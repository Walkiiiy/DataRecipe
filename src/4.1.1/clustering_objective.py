"""Clustering objective evaluator based on density and separation.

Formulae:
    Dens(C_k) = sum(||x - c_k||) / (max(||x - c_k||) + eps)
    Sep(C_k)  = average_j!=k ||c_k - c_j||, or 0 when only one cluster
    J(C)      = sum_k Dens(C_k) * Sep(C_k)
"""

from __future__ import annotations

from dataclasses import dataclass
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


if __name__ == "__main__":
    # Minimal sanity example.
    clusters_demo = [
        np.array([[0.0, 0.0], [0.2, 0.1], [0.1, -0.1]], dtype=np.float64),
        np.array([[3.0, 3.0], [3.2, 3.1]], dtype=np.float64),
    ]
    evaluator = ObjectiveEvaluator()
    print(f"J = {evaluator.evaluate(clusters_demo):.6f}")
