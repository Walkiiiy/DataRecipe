"""Clustering evaluators for capability-space construction.

4.1.2 uses the Lemma-1 split gate:
    R(C)      = mean(||x - mu||^2) / N
    gain      = beta * (m^(-2/d) - (m + k - 1)^(-2/d))
    penalty   = max_j R(C_j) - R(P)
    split iff penalty < gain

The legacy density/separation objective is kept for backward compatibility
with 4.1.1-era imports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


ArrayLike = np.ndarray | list[list[float]] | list[np.ndarray]


@dataclass
class ObjectiveEvaluator:
    """Evaluate clustering objective J from a list of cluster vectors."""

    eps: float = 1e-5

    def _to_2d_array(self, cluster: ArrayLike) -> np.ndarray:
        arr = np.asarray(cluster, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("Each cluster must be a 2D array-like of shape [n_samples, n_features].")
        if arr.shape[0] == 0:
            raise ValueError("Cluster must not be empty.")
        return arr

    def _normalize_clusters(self, clusters: Iterable[ArrayLike]) -> list[np.ndarray]:
        cluster_list = [self._to_2d_array(c) for c in clusters]
        if not cluster_list:
            return []

        feat_dim = cluster_list[0].shape[1]
        for i, c in enumerate(cluster_list):
            if c.shape[1] != feat_dim:
                raise ValueError(
                    f"Feature dimension mismatch at cluster {i}: expected {feat_dim}, got {c.shape[1]}."
                )
        return cluster_list

    def density(self, cluster: ArrayLike) -> float:
        """Compute Dens(C_k) for one cluster."""
        # Handle empty clusters explicitly (requested behavior): density = 0.0.
        c = np.asarray(cluster, dtype=np.float64)
        if c.size == 0:
            return 0.0
        if c.ndim == 1:
            c = c.reshape(1, -1)
        if c.ndim != 2:
            raise ValueError("Each cluster must be a 2D array-like of shape [n_samples, n_features].")

        num_items = int(c.shape[0])
        if num_items == 0:
            return 0.0
        if num_items == 1:
            # mean_dist = 0, so Dens = 1 * (2 - 0) = 2.0
            return 2.0

        center = c.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(c - center, axis=1)
        mean_dist = float(np.mean(distances))
        return float(num_items * (2.0 - mean_dist))

    def separation(self, centers: np.ndarray, index: int) -> float:
        """Compute Sep(C_k) from all cluster centers."""
        num_clusters = centers.shape[0]
        if num_clusters <= 1:
            return 0.0

        dists = np.linalg.norm(centers - centers[index], axis=1)
        # Remove self-distance (0) before averaging.
        dists_others = np.delete(dists, index)
        return float(np.mean(dists_others))

    def evaluate(self, clusters: Iterable[ArrayLike]) -> float:
        """Return global objective J(C)."""
        cluster_list = self._normalize_clusters(clusters)
        num_clusters = len(cluster_list)

        if num_clusters <= 1:
            return 0.0

        centers = np.stack([c.mean(axis=0) for c in cluster_list], axis=0)
        densities = np.asarray([self.density(c) for c in cluster_list], dtype=np.float64)
        separations = np.asarray([self.separation(centers, i) for i in range(num_clusters)], dtype=np.float64)
        return float(np.sum(densities * separations))


@dataclass
class ClusterRiskStats:
    """Geometry-derived risk quantities for one cluster."""

    sample_count: int
    total_sse: float
    mean_sq_radius: float
    structural_risk_density: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class LemmaSplitDecision:
    """Decision details for the Lemma-1 split gate."""

    accepted: bool
    basis_count_before: int
    basis_count_after: int
    split_k: int
    intrinsic_dim: float
    split_gain_scale: float
    parent_risk_density: float
    child_risk_densities: list[float]
    max_child_risk_density: float
    risk_delta: float
    bias_gain: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class CapabilitySplitEvaluator:
    """Evaluate structural risk and Lemma-1 split acceptance."""

    eps: float = 1e-12

    @staticmethod
    def _to_2d_array(vectors: ArrayLike) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("Cluster vectors must be a 2D array-like of shape [n_samples, n_features].")
        if arr.shape[0] == 0:
            raise ValueError("Cluster vectors must not be empty.")
        return arr

    def risk_stats(self, vectors: ArrayLike) -> ClusterRiskStats:
        """Return SSE, mean squared radius, and structural risk density.

        The document defines I(C) as the mean squared radius, not sklearn's
        total inertia. The structural risk density is therefore I(C) / N.
        """
        arr = self._to_2d_array(vectors)
        n = int(arr.shape[0])
        center = arr.mean(axis=0, keepdims=True)
        total_sse = float(np.sum(np.square(arr - center)))
        mean_sq_radius = total_sse / max(1, n)
        structural_risk_density = mean_sq_radius / max(1, n)
        return ClusterRiskStats(
            sample_count=n,
            total_sse=total_sse,
            mean_sq_radius=float(mean_sq_radius),
            structural_risk_density=float(structural_risk_density),
        )

    def evaluate_lemma_gate(
        self,
        parent_vectors: ArrayLike,
        child_vectors: Iterable[ArrayLike],
        basis_count_before: int,
        intrinsic_dim: float,
        split_gain_scale: float,
    ) -> LemmaSplitDecision:
        """Evaluate whether one already-selected split candidate passes Lemma 1."""
        children = [self._to_2d_array(c) for c in child_vectors]
        if len(children) < 2:
            raise ValueError("A split candidate must have at least two child clusters.")

        parent_stats = self.risk_stats(parent_vectors)
        child_stats = [self.risk_stats(c) for c in children]
        child_risks = [float(s.structural_risk_density) for s in child_stats]
        max_child_risk = float(max(child_risks))
        risk_delta = float(max_child_risk - parent_stats.structural_risk_density)

        m_before = max(1, int(basis_count_before))
        split_k = len(children)
        m_after = max(m_before + split_k - 1, m_before)
        d = max(float(intrinsic_dim), self.eps)
        beta = max(float(split_gain_scale), 0.0)
        exponent = -2.0 / d
        bias_gain = float(beta * ((m_before**exponent) - (m_after**exponent)))

        return LemmaSplitDecision(
            accepted=bool(risk_delta < bias_gain),
            basis_count_before=m_before,
            basis_count_after=m_after,
            split_k=split_k,
            intrinsic_dim=float(d),
            split_gain_scale=float(beta),
            parent_risk_density=float(parent_stats.structural_risk_density),
            child_risk_densities=child_risks,
            max_child_risk_density=max_child_risk,
            risk_delta=risk_delta,
            bias_gain=bias_gain,
        )


if __name__ == "__main__":
    # Minimal sanity example.
    clusters_demo = [
        np.array([[0.0, 0.0], [0.2, 0.1], [0.1, -0.1]], dtype=np.float64),
        np.array([[3.0, 3.0], [3.2, 3.1]], dtype=np.float64),
    ]
    evaluator = ObjectiveEvaluator()
    print(f"J = {evaluator.evaluate(clusters_demo):.6f}")
