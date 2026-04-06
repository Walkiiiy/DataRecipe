#!/usr/bin/env python3
"""Online Recipe Evolution (4.3).

本文件实现论文复现所需的核心类 ``RecipeEvolver``，对应以下四个阶段：
1) 计算锚点梯度
2) 更新能力权重 beta
3) 全局打分并按玻尔兹曼分布采样 batch
4) 计算奖励并更新策略权重 alpha

实现重点（工程约束）：
- 只抽取指定层（默认 ``lm_head``）梯度，不额外保存/统计全模型梯度。
- 阶段 4 的单样本梯度使用 ``torch.func.vmap + torch.func.grad``，避免逐样本 backward 循环。
- alpha / beta 更新全部放在 ``torch.no_grad()`` 下，和大模型训练计算图解耦。
- 支持通过评分文件路径（如 ``data/dialogsum/score/pdm_scored.jsonl``）构建 ``E_matrix (N,k,m)``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import functional_call, grad, vmap


TensorDict = Dict[str, torch.Tensor]


def _load_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """读取 json / jsonl 并返回对象列表。

    注意：为了避免 splitlines() 把某些 Unicode 行分隔符误拆，jsonl 用 split("\\n")。
    """
    fp = Path(path)
    text = fp.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        rows: List[Dict[str, Any]] = []
        for line_no, raw in enumerate(text.split("\n"), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {fp}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Invalid JSONL object at {fp}:{line_no}: expected dict.")
            rows.append(row)
        return rows

    if not isinstance(obj, list):
        raise ValueError(f"JSON file must be a list: {fp}")
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(obj):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid JSON object at {fp}[{idx}]: expected dict.")
        out.append(row)
    return out


def _choose_row_id(row: Dict[str, Any], fallback_idx: int, preferred_key: str = "id") -> Any:
    """从一条样本记录中选择稳定 ID。"""
    if preferred_key and preferred_key in row and row.get(preferred_key) is not None:
        return row.get(preferred_key)
    for key in ("id", "data_id", "uid", "idx", "index"):
        if key in row and row.get(key) is not None:
            return row.get(key)
    return fallback_idx


def _parse_sparse_or_dense_vector(value: Any) -> Dict[int, float]:
    """把 list / dict 形式向量统一成稀疏字典 `{dim_idx: value}`。"""
    out: Dict[int, float] = {}
    if isinstance(value, list):
        for idx, v in enumerate(value):
            try:
                fv = float(v)
            except Exception:  # noqa: BLE001
                continue
            if fv != 0.0:
                out[int(idx)] = fv
        return out

    if isinstance(value, dict):
        for k, v in value.items():
            try:
                idx = int(k)
                fv = float(v)
            except Exception:  # noqa: BLE001
                continue
            if idx >= 0 and fv != 0.0:
                out[idx] = fv
        return out
    return out


def _infer_loss_from_outputs(outputs: Any, batch: Mapping[str, Any]) -> torch.Tensor:
    """通用 loss 推断逻辑。

    优先使用 HuggingFace 风格 `outputs.loss`；否则尝试 logits + labels 的 CE。
    """
    if hasattr(outputs, "loss") and outputs.loss is not None:
        return outputs.loss

    labels = batch.get("labels", None)
    if labels is None:
        raise ValueError("Cannot infer loss: batch does not contain `labels` and model output has no `.loss`.")
    if not torch.is_tensor(labels):
        raise ValueError("`labels` must be a tensor when inferring loss from logits.")

    logits = None
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
        logits = outputs[0]

    if logits is None or not torch.is_tensor(logits):
        raise ValueError("Cannot infer loss: output has neither `.loss` nor tensor `logits`.")

    if logits.dim() == 2 and labels.dim() == 1:
        return F.cross_entropy(logits, labels)
    if logits.dim() == 3 and labels.dim() == 2:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
    raise ValueError(
        "Unsupported logits/labels shapes for default CE loss: "
        f"logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
    )


@dataclass
class StepStats:
    """单步演化返回统计。"""

    step: int
    train_loss: float
    avg_reward: float
    sampled_indices: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    utilities: torch.Tensor


class RecipeEvolver:
    """Online Recipe Evolution 主类。

    参数说明（核心张量）：
    - ``E_matrix``: `(N, k, m)`，全量数据静态多策略能力向量评分。
    - ``top_k_indices``: `(N, K_max)`，每条数据对应候选能力簇索引。
    - ``alpha``: `(k,)`，策略权重，默认均匀初始化。
    - ``beta``: `(m,)`，能力维度权重，默认均匀初始化。

    其中 `alpha/beta` 只作为外部状态，不参与模型参数计算图。
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        pool_batch: Mapping[str, torch.Tensor],
        E_matrix: torch.Tensor,
        top_k_indices: torch.Tensor,
        layer_name: str = "lm_head",
        alpha: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        eta_beta: float = 0.1,
        gamma_alpha: float = 0.8,
        epsilon: float = 0.05,
        gamma_T: float = 1.0,
        frequency_penalty: float = 0.1,
        anchor_ema_momentum: float = 0.8,
        prune_patience: int = 3,
        prune_reward_threshold: float = -0.05,
        mapper_utility_mode: str = "beta_weighted",
        loss_fn: Optional[Callable[[Any, Mapping[str, Any]], torch.Tensor]] = None,
        score_device: Optional[torch.device | str] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.layer_name = layer_name
        self.eta_beta = float(eta_beta)
        self.gamma_alpha = float(gamma_alpha)
        self.epsilon = float(epsilon)
        self.gamma_T = float(gamma_T)
        self.frequency_penalty = max(0.0, float(frequency_penalty))
        self.anchor_ema_momentum = float(anchor_ema_momentum)
        self.prune_patience = int(prune_patience)
        self.prune_reward_threshold = float(prune_reward_threshold)
        self.mapper_utility_mode = str(mapper_utility_mode)
        self.loss_fn = loss_fn if loss_fn is not None else _infer_loss_from_outputs

        self.model_device = next(self.model.parameters()).device
        self.score_device = (
            torch.device(score_device)
            if score_device is not None
            else (E_matrix.device if torch.is_tensor(E_matrix) else torch.device("cpu"))
        )

        if E_matrix.ndim != 3:
            raise ValueError(f"E_matrix must be 3D (N,k,m), got shape={tuple(E_matrix.shape)}")
        if top_k_indices.ndim != 2:
            raise ValueError(
                f"top_k_indices must be 2D (N,K_max), got shape={tuple(top_k_indices.shape)}"
            )
        if E_matrix.size(0) != top_k_indices.size(0):
            raise ValueError("E_matrix and top_k_indices must share the same N.")

        self.E_matrix = E_matrix.detach().to(self.score_device, dtype=torch.float32)
        self.top_k_indices = top_k_indices.detach().to(self.score_device, dtype=torch.long)
        self.N, self.k, self.m = self.E_matrix.shape
        self.K_max = self.top_k_indices.size(1)

        # pool_batch 是模型训练所需的全量输入池（第一维长度应为 N）。
        self.pool_batch: TensorDict = {}
        for key, value in pool_batch.items():
            if not torch.is_tensor(value):
                raise TypeError(f"pool_batch[{key}] must be tensor, got {type(value)}")
            if value.dim() == 0:
                raise ValueError(f"pool_batch[{key}] must be batched tensor with dim>=1.")
            if value.size(0) != self.N:
                raise ValueError(
                    f"pool_batch[{key}] first dim must be N={self.N}, got {value.size(0)}."
                )
            self.pool_batch[key] = value

        with torch.no_grad():
            if alpha is None:
                self.alpha = torch.full(
                    (self.k,),
                    1.0 / max(1, self.k),
                    dtype=torch.float32,
                    device=self.score_device,
                )
            else:
                if alpha.shape != (self.k,):
                    raise ValueError(f"alpha shape must be ({self.k},), got {tuple(alpha.shape)}")
                a = alpha.detach().to(self.score_device, dtype=torch.float32).clamp_min(0.0)
                self.alpha = a / a.sum().clamp_min(1e-12)

            if beta is None:
                self.beta = torch.full(
                    (self.m,),
                    1.0 / max(1, self.m),
                    dtype=torch.float32,
                    device=self.score_device,
                )
            else:
                if beta.shape != (self.m,):
                    raise ValueError(f"beta shape must be ({self.m},), got {tuple(beta.shape)}")
                b = beta.detach().to(self.score_device, dtype=torch.float32).clamp_min(0.0)
                self.beta = b / b.sum().clamp_min(1e-12)

        self._target_param_names, self._target_params = self._get_target_layer_params()
        self.gradient_dim = int(sum(p.numel() for p in self._target_params))

        # 预先建立 capability -> 样本索引表，用于锚点采样。
        self.capability_to_indices: List[torch.Tensor] = self._build_capability_index()

        # EMA 锚点梯度状态（位于模型设备，供 beta 更新与奖励计算复用）。
        self._running_anchor_grads: Optional[torch.Tensor] = None

        # 动态修剪状态（位于 score_device，与打分/采样同设备）。
        self.bad_strike_counter = torch.zeros((self.N,), dtype=torch.long, device=self.score_device)
        self.pruned_mask = torch.zeros((self.N,), dtype=torch.bool, device=self.score_device)
        # 采样频次计数（位于 score_device），用于频次惩罚避免 mode collapse。
        self.sample_counts = torch.zeros((self.N,), dtype=torch.long, device=self.score_device)

        self.step_id = 0
        self.alpha_history: List[torch.Tensor] = [self.alpha.detach().clone().cpu()]
        self.beta_history: List[torch.Tensor] = [self.beta.detach().clone().cpu()]

    # -----------------------------
    # 构建器：从评分文件路径生成 E_matrix / top_k_indices
    # -----------------------------
    @classmethod
    def from_score_files(
        cls,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        pool_batch: Mapping[str, torch.Tensor],
        score_file_paths: Sequence[str | Path],
        top_k_file_path: Optional[str | Path] = None,
        id_field: str = "id",
        vector_field: str = "mapped_vector",
        fallback_vector_field: str = "score",
        top_k_field: str = "top_k_indices",
        m: Optional[int] = None,
        K_max: Optional[int] = None,
        layer_name: str = "lm_head",
        alpha: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        eta_beta: float = 0.1,
        gamma_alpha: float = 0.8,
        epsilon: float = 0.05,
        gamma_T: float = 1.0,
        anchor_ema_momentum: float = 0.8,
        prune_patience: int = 3,
        prune_reward_threshold: float = -0.05,
        mapper_utility_mode: str = "beta_weighted",
        loss_fn: Optional[Callable[[Any, Mapping[str, Any]], torch.Tensor]] = None,
        score_device: Optional[torch.device | str] = None,
    ) -> "RecipeEvolver":
        """通过可插拔评分文件路径构建 RecipeEvolver。

        典型场景：将 `score/*.jsonl` 作为不同 mapper 输入，自动拼出 `E_matrix (N,k,m)`。
        """
        E_matrix, top_k_indices = cls.build_feature_tensors_from_score_files(
            score_file_paths=score_file_paths,
            top_k_file_path=top_k_file_path,
            id_field=id_field,
            vector_field=vector_field,
            fallback_vector_field=fallback_vector_field,
            top_k_field=top_k_field,
            m=m,
            K_max=K_max,
            device=score_device,
        )
        return cls(
            model=model,
            optimizer=optimizer,
            pool_batch=pool_batch,
            E_matrix=E_matrix,
            top_k_indices=top_k_indices,
            layer_name=layer_name,
            alpha=alpha,
            beta=beta,
            eta_beta=eta_beta,
            gamma_alpha=gamma_alpha,
            epsilon=epsilon,
            gamma_T=gamma_T,
            anchor_ema_momentum=anchor_ema_momentum,
            prune_patience=prune_patience,
            prune_reward_threshold=prune_reward_threshold,
            mapper_utility_mode=mapper_utility_mode,
            loss_fn=loss_fn,
            score_device=score_device,
        )

    @staticmethod
    def build_feature_tensors_from_score_files(
        *,
        score_file_paths: Sequence[str | Path],
        top_k_file_path: Optional[str | Path] = None,
        id_field: str = "id",
        vector_field: str = "mapped_vector",
        fallback_vector_field: str = "score",
        top_k_field: str = "top_k_indices",
        m: Optional[int] = None,
        K_max: Optional[int] = None,
        device: Optional[torch.device | str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从多个 mapper 的评分文件构建 `(E_matrix, top_k_indices)`。

        输出 shape：
        - E_matrix: `(N, k, m)`
        - top_k_indices: `(N, K_max)`
        """
        if not score_file_paths:
            raise ValueError("score_file_paths must not be empty.")

        mapper_rows: List[List[Dict[str, Any]]] = [
            _load_json_or_jsonl(path) for path in score_file_paths
        ]
        if not mapper_rows[0]:
            raise ValueError(f"Empty score file: {score_file_paths[0]}")

        # 使用第一个 mapper 的顺序作为主顺序，后续 mapper 按 id 对齐。
        row_ids: List[str] = []
        first_rows = mapper_rows[0]
        for idx, row in enumerate(first_rows):
            rid = _choose_row_id(row, idx, preferred_key=id_field)
            row_ids.append(str(rid))
        N = len(row_ids)
        k = len(mapper_rows)

        row_maps: List[Dict[str, Dict[str, Any]]] = []
        for rows in mapper_rows:
            mp: Dict[str, Dict[str, Any]] = {}
            for idx, row in enumerate(rows):
                rid = _choose_row_id(row, idx, preferred_key=id_field)
                mp[str(rid)] = row
            row_maps.append(mp)

        # 自动推断 m：综合所有 mapper 的向量长度和 top_k 维度索引上界。
        inferred_m = 0
        for rows in mapper_rows:
            for row in rows:
                vec_sparse = _parse_sparse_or_dense_vector(row.get(vector_field))
                if not vec_sparse:
                    vec_sparse = _parse_sparse_or_dense_vector(row.get(fallback_vector_field))
                if vec_sparse:
                    inferred_m = max(inferred_m, max(vec_sparse.keys()) + 1)
                tk = row.get(top_k_field)
                if isinstance(tk, list):
                    for x in tk:
                        try:
                            idx_i = int(x)
                        except Exception:  # noqa: BLE001
                            continue
                        if idx_i >= 0:
                            inferred_m = max(inferred_m, idx_i + 1)
        if m is None:
            if inferred_m <= 0:
                raise ValueError("Failed to infer capability dimension m from score files.")
            m = inferred_m
        m = int(m)
        if m <= 0:
            raise ValueError(f"m must be positive, got {m}")

        # 构建 E_matrix
        score_device = torch.device(device) if device is not None else torch.device("cpu")
        E_matrix = torch.zeros((N, k, m), dtype=torch.float32, device=score_device)
        for mapper_idx, mp in enumerate(row_maps):
            for n_idx, rid in enumerate(row_ids):
                row = mp.get(rid)
                if row is None:
                    # 某 mapper 缺失该样本时，保持全 0。
                    continue
                vec_sparse = _parse_sparse_or_dense_vector(row.get(vector_field))
                if not vec_sparse:
                    vec_sparse = _parse_sparse_or_dense_vector(row.get(fallback_vector_field))
                for dim_idx, value in vec_sparse.items():
                    if 0 <= dim_idx < m:
                        E_matrix[n_idx, mapper_idx, dim_idx] = float(value)

        # top_k 来源优先级：显式 top_k_file_path > 第一份 score 文件内字段。
        if top_k_file_path is not None:
            top_k_rows = _load_json_or_jsonl(top_k_file_path)
            top_k_map: Dict[str, Dict[str, Any]] = {}
            for idx, row in enumerate(top_k_rows):
                rid = _choose_row_id(row, idx, preferred_key=id_field)
                top_k_map[str(rid)] = row
        else:
            top_k_map = row_maps[0]

        if K_max is None:
            inferred_kmax = 0
            for rid in row_ids:
                row = top_k_map.get(rid)
                if row is None:
                    continue
                tk = row.get(top_k_field)
                if isinstance(tk, list):
                    inferred_kmax = max(inferred_kmax, len(tk))
            if inferred_kmax <= 0:
                raise ValueError(
                    "Failed to infer K_max. Please provide top_k_file_path or K_max explicitly."
                )
            K_max = inferred_kmax
        K_max = int(K_max)
        if K_max <= 0:
            raise ValueError(f"K_max must be positive, got {K_max}")

        top_k_indices = torch.full(
            (N, K_max),
            fill_value=-1,
            dtype=torch.long,
            device=score_device,
        )
        for n_idx, rid in enumerate(row_ids):
            row = top_k_map.get(rid)
            if row is None:
                continue
            tk = row.get(top_k_field)
            if not isinstance(tk, list):
                continue
            for t, x in enumerate(tk[:K_max]):
                try:
                    idx_i = int(x)
                except Exception:  # noqa: BLE001
                    continue
                if 0 <= idx_i < m:
                    top_k_indices[n_idx, t] = idx_i

        return E_matrix, top_k_indices

    # -----------------------------
    # 阶段 1：锚点梯度
    # -----------------------------
    def _get_target_layer_params(self) -> Tuple[List[str], List[torch.Tensor]]:
        """获取指定层参数（如 lm_head.*），用于梯度抽取。"""
        names: List[str] = []
        params: List[torch.Tensor] = []
        prefix = f"{self.layer_name}."
        for name, p in self.model.named_parameters():
            if name == self.layer_name or name.startswith(prefix):
                if p.requires_grad:
                    names.append(name)
                    params.append(p)
        if not names:
            raise ValueError(f"No trainable parameters found for layer_name='{self.layer_name}'.")
        return names, params

    def _build_capability_index(self) -> List[torch.Tensor]:
        """预构建能力簇 -> 样本索引映射，提升锚点采样效率。"""
        out: List[torch.Tensor] = []
        for j in range(self.m):
            # 标记样本是否在 top_k 候选中包含能力 j
            mask = (self.top_k_indices == j).any(dim=1)
            idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
            out.append(idx)
        return out

    def sample_anchor_indices(
        self,
        anchor_size_per_capability: int,
        *,
        replacement_if_needed: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[int, torch.Tensor]:
        """从每个能力簇采样锚点样本索引。

        返回 dict: `capability_idx -> Tensor[num_anchor]`（在 score_device 上）。
        """
        if anchor_size_per_capability <= 0:
            raise ValueError("anchor_size_per_capability must be > 0.")

        out: Dict[int, torch.Tensor] = {}
        n_anchor = int(anchor_size_per_capability)
        for j in range(self.m):
            cand = self.capability_to_indices[j]
            if cand.numel() == 0:
                out[j] = torch.empty((0,), dtype=torch.long, device=self.score_device)
                continue
            if cand.numel() >= n_anchor:
                perm = torch.randperm(cand.numel(), device=self.score_device, generator=generator)
                out[j] = cand[perm[:n_anchor]]
                continue
            if not replacement_if_needed:
                out[j] = cand
                continue
            # 候选不足时允许有放回采样，保证每个能力都能构建锚点集。
            pick = torch.randint(
                low=0,
                high=cand.numel(),
                size=(n_anchor,),
                device=self.score_device,
                generator=generator,
            )
            out[j] = cand[pick]
        return out

    def _gather_pool_batch(self, indices: torch.Tensor) -> TensorDict:
        """按样本索引从全量池切 batch，并移动到模型设备。"""
        if indices.device != self.pool_batch[next(iter(self.pool_batch))].device:
            # pool_batch 往往在 CPU，先把索引移到对应设备再 index_select。
            idx_for_pool = indices.to(self.pool_batch[next(iter(self.pool_batch))].device)
        else:
            idx_for_pool = indices

        out: TensorDict = {}
        for key, value in self.pool_batch.items():
            selected = value.index_select(0, idx_for_pool)
            out[key] = selected.to(self.model_device, non_blocking=True)
        return out

    def extract_layer_gradients(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """只抽取指定层梯度并展平为一维向量。

        说明：
        - 这里使用 `torch.autograd.grad(loss, target_layer_params)`，
          只请求目标层梯度，不把全模型梯度写入 `.grad`。
        - 该函数用于锚点梯度和其他诊断，不承担参数更新。
        """
        outputs = self.model(**batch)
        loss = self.loss_fn(outputs, batch)

        grads = torch.autograd.grad(
            loss,
            self._target_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        flat_grads: List[torch.Tensor] = []
        for p, g in zip(self._target_params, grads):
            if g is None:
                flat_grads.append(torch.zeros_like(p).reshape(-1))
            else:
                flat_grads.append(g.detach().reshape(-1))
        return torch.cat(flat_grads, dim=0)

    def compute_anchor_gradients(
        self,
        anchor_indices_by_capability: Mapping[int, torch.Tensor],
        *,
        chunk_size: int = 8,
    ) -> torch.Tensor:
        """阶段 1：计算每个能力簇的平均锚点梯度。

        输出 shape: `(m, gradient_dim)`。
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        anchor_grads = torch.zeros(
            (self.m, self.gradient_dim),
            dtype=torch.float32,
            device=self.model_device,
        )

        for j in range(self.m):
            idx = anchor_indices_by_capability.get(j, None)
            if idx is None or idx.numel() == 0:
                continue

            idx = idx.to(self.score_device, dtype=torch.long).reshape(-1)
            total = 0
            g_sum = torch.zeros((self.gradient_dim,), dtype=torch.float32, device=self.model_device)

            for start in range(0, idx.numel(), chunk_size):
                part = idx[start : start + chunk_size]
                batch = self._gather_pool_batch(part)
                g_part = self.extract_layer_gradients(batch)  # 当前 chunk 的均值 loss 梯度
                bs = int(part.numel())
                g_sum = g_sum + g_part * float(bs)
                total += bs

            if total > 0:
                anchor_grads[j] = g_sum / float(total)

        return anchor_grads

    def update_running_anchor_gradients(
        self,
        current_anchor_gradients: torch.Tensor,
        *,
        momentum: Optional[float] = None,
    ) -> torch.Tensor:
        """用 EMA 更新锚点梯度状态。

        公式：
        - 首次：running = current
        - 否则：running = momentum * running + (1 - momentum) * current
        """
        mom = self.anchor_ema_momentum if momentum is None else float(momentum)
        mom = min(max(mom, 0.0), 0.999999)

        with torch.no_grad():
            cur = current_anchor_gradients.detach().to(self.model_device, dtype=torch.float32)
            if self._running_anchor_grads is None:
                self._running_anchor_grads = cur.clone()
            else:
                self._running_anchor_grads = mom * self._running_anchor_grads + (1.0 - mom) * cur
            return self._running_anchor_grads.detach().clone()

    def get_running_anchor_gradients(self) -> Optional[torch.Tensor]:
        """返回当前 EMA 锚点梯度（若尚未初始化则返回 None）。"""
        if self._running_anchor_grads is None:
            return None
        return self._running_anchor_grads.detach().clone()

    def update_pruning_state(
        self,
        *,
        batch_indices: torch.Tensor,
        rewards: torch.Tensor,
        prune_patience: Optional[int] = None,
        prune_reward_threshold: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据 batch 奖励更新动态修剪状态。

        返回：
        - bad_strike_counter: `(N,)`
        - pruned_mask: `(N,)`
        """
        patience = int(self.prune_patience if prune_patience is None else prune_patience)
        threshold = float(self.prune_reward_threshold if prune_reward_threshold is None else prune_reward_threshold)
        patience = max(1, patience)

        with torch.no_grad():
            idx = batch_indices.detach().to(self.score_device, dtype=torch.long).reshape(-1)
            rw = rewards.detach().to(self.score_device, dtype=torch.float32).reshape(-1)
            if idx.numel() == 0:
                return self.bad_strike_counter, self.pruned_mask
            if idx.numel() != rw.numel():
                raise ValueError(
                    f"batch_indices and rewards length mismatch: {idx.numel()} vs {rw.numel()}"
                )

            # good 样本连续计数清零
            good_idx = idx[rw >= threshold]
            if good_idx.numel() > 0:
                good_idx = torch.unique(good_idx)
                self.bad_strike_counter[good_idx] = 0

            # bad 样本连续计数 +1（支持重复索引）
            bad_idx = idx[rw < threshold]
            if bad_idx.numel() > 0:
                inc = torch.bincount(bad_idx, minlength=self.N)
                if inc.numel() > self.N:
                    inc = inc[: self.N]
                self.bad_strike_counter = self.bad_strike_counter + inc.to(self.bad_strike_counter.dtype)

            self.pruned_mask = self.pruned_mask | (self.bad_strike_counter >= patience)
            return self.bad_strike_counter.detach().clone(), self.pruned_mask.detach().clone()

    # -----------------------------
    # 阶段 2：beta 更新
    # -----------------------------
    def update_beta(
        self,
        anchor_gradients: torch.Tensor,
        *,
        eta_beta: Optional[float] = None,
    ) -> torch.Tensor:
        """阶段 2：依据锚点梯度范数更新 beta，并做 L1 归一化。"""
        lr = float(self.eta_beta if eta_beta is None else eta_beta)
        with torch.no_grad():
            norms = anchor_gradients.detach().norm(p=2, dim=1)  # (m,)
            norm_sum = norms.sum()
            if torch.isfinite(norm_sum) and float(norm_sum.item()) > 0.0:
                inc = lr * (norms / norm_sum).to(self.score_device)
                self.beta = self.beta + inc
            # L1 归一化
            self.beta = self.beta.clamp_min(0.0)
            self.beta = self.beta / self.beta.sum().clamp_min(1e-12)
            return self.beta.detach().clone()

    # -----------------------------
    # 阶段 3：全局打分 + 玻尔兹曼采样
    # -----------------------------
    def compute_global_scores(self) -> torch.Tensor:
        """计算 `S(x) = alpha^T E_x beta`，输出 `(N,)`。"""
        with torch.no_grad():
            scores = torch.einsum("k,nkm,m->n", self.alpha, self.E_matrix, self.beta)
            # 对已修剪样本施加极小分数，令其采样概率趋近于 0。
            scores = scores.masked_fill(self.pruned_mask, -1e9)
        return scores

    def score_and_sample_batch(
        self,
        batch_size: int,
        *,
        gamma_T: Optional[float] = None,
        replacement: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """阶段 3：根据玻尔兹曼分布采样 batch 索引。

        返回：
        - sampled_indices: `(B,)`
        - probs: `(N,)`
        - scores: `(N,)`
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

        tau = float(self.gamma_T if gamma_T is None else gamma_T)
        tau = max(tau, 1e-8)

        with torch.no_grad():
            active_count = int((~self.pruned_mask).sum().item())
            if active_count <= 0:
                raise RuntimeError("All samples are pruned; cannot sample batch.")

            scores = self.compute_global_scores()
            logits = (scores / tau) - (
                self.frequency_penalty * self.sample_counts.to(dtype=scores.dtype)
            )
            logits = logits - logits.max()
            probs = torch.softmax(logits, dim=0)

            bsz = int(batch_size)
            if not replacement:
                bsz = min(bsz, active_count)
            sampled = torch.multinomial(probs, num_samples=bsz, replacement=replacement)
            if sampled.numel() > 0:
                inc = torch.bincount(sampled, minlength=self.N)
                if inc.numel() > self.N:
                    inc = inc[: self.N]
                self.sample_counts = self.sample_counts + inc.to(self.sample_counts.dtype)
            return sampled, probs, scores

    # -----------------------------
    # 阶段 4：奖励计算 + alpha 更新
    # -----------------------------
    def _split_dynamic_batch_fields(self, batch: Mapping[str, torch.Tensor]) -> Tuple[List[str], List[torch.Tensor]]:
        """把 batch 中沿第 0 维可 vmapped 的字段提取出来。"""
        keys: List[str] = []
        vals: List[torch.Tensor] = []
        if not batch:
            return keys, vals
        bsz = next(iter(batch.values())).size(0)
        for k, v in batch.items():
            if not torch.is_tensor(v):
                continue
            if v.dim() == 0:
                continue
            if v.size(0) != bsz:
                continue
            keys.append(k)
            vals.append(v)
        return keys, vals

    def compute_per_sample_layer_gradients_vmap(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """使用 `torch.func.vmap + grad` 计算单样本梯度（指定层）。

        输出 shape: `(B, gradient_dim)`。
        """
        dyn_keys, dyn_vals = self._split_dynamic_batch_fields(batch)
        if not dyn_keys:
            raise ValueError("No vmappable batch fields found (empty dynamic batch).")

        target_param_tuple = tuple(self._target_params)

        def single_loss(layer_params: Tuple[torch.Tensor, ...], *sample_vals: torch.Tensor) -> torch.Tensor:
            # 只替换目标层参数，其他层参数沿用当前 model（strict=False）。
            override = {
                name: p for name, p in zip(self._target_param_names, layer_params)
            }
            kwargs: Dict[str, Any] = {}
            for key, sv in zip(dyn_keys, sample_vals):
                kwargs[key] = sv.unsqueeze(0)

            outputs = functional_call(
                self.model,
                override,
                args=(),
                kwargs=kwargs,
                strict=False,
            )
            return self.loss_fn(outputs, kwargs)

        grad_fn = grad(single_loss)
        in_dims = (None,) + tuple(0 for _ in dyn_vals)

        # 某些模型包含 dropout 等随机算子时，需要声明 randomness 行为。
        try:
            per_param_grads = vmap(grad_fn, in_dims=in_dims, randomness="different", chunk_size=2)(
                target_param_tuple, *dyn_vals
            )
        except TypeError:
            per_param_grads = vmap(grad_fn, in_dims=in_dims)(target_param_tuple, *dyn_vals)

        flat_parts: List[torch.Tensor] = []
        bsz = dyn_vals[0].size(0)
        for g in per_param_grads:
            flat_parts.append(g.reshape(bsz, -1))
        return torch.cat(flat_parts, dim=1)

    def compute_local_rewards(
        self,
        *,
        batch_indices: torch.Tensor,
        per_sample_grads: torch.Tensor,
        anchor_gradients: torch.Tensor,
    ) -> torch.Tensor:
        """按候选能力集合 K_x 计算局部奖励 R(x)。"""
        g_x = F.normalize(per_sample_grads.detach(), p=2, dim=1, eps=1e-12)  # (B, G)
        g_c = F.normalize(anchor_gradients.detach(), p=2, dim=1, eps=1e-12)  # (m, G)

        topk = self.top_k_indices[batch_indices.to(self.score_device)]  # (B, K_max)
        topk = topk.to(self.model_device)
        valid = (topk >= 0) & (topk < self.m)
        topk_safe = topk.clamp(min=0, max=self.m - 1)

        g_sel = g_c[topk_safe]  # (B, K_max, G)
        sim = (g_x.unsqueeze(1) * g_sel).sum(dim=-1)  # (B, K_max)
        sim = sim * valid.to(sim.dtype)
        denom = valid.sum(dim=1).clamp_min(1).to(sim.dtype)
        rewards = sim.sum(dim=1) / denom
        return rewards.detach()

    def compute_mapper_utilities(
        self,
        *,
        batch_indices: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """计算策略预期效用 U_i。

        注：题设中 `v_{x,i}` 是 m 维向量，而 U_i 需要标量。
        默认实现使用 `beta` 对 `v_{x,i}` 做加权压缩：
        `v_scalar(x,i) = <v_{x,i}, beta>`。
        """
        E_b = self.E_matrix[batch_indices.to(self.score_device)]  # (B, k, m)
        if self.mapper_utility_mode == "beta_weighted":
            v_scalar = torch.einsum("bkm,m->bk", E_b, self.beta)  # (B, k)
        elif self.mapper_utility_mode == "mean":
            v_scalar = E_b.mean(dim=-1)
        elif self.mapper_utility_mode == "sum":
            v_scalar = E_b.sum(dim=-1)
        else:
            raise ValueError(
                "Unsupported mapper_utility_mode. "
                f"Expected one of ['beta_weighted','mean','sum'], got {self.mapper_utility_mode!r}."
            )

        r = rewards.detach().to(self.score_device).unsqueeze(1)  # (B,1)
        U = (r * v_scalar).mean(dim=0)  # (k,)
        return U.detach()

    def update_alpha(
        self,
        utilities: torch.Tensor,
        *,
        gamma_alpha: Optional[float] = None,
        epsilon: Optional[float] = None,
    ) -> torch.Tensor:
        """指数更新 alpha，并加入 epsilon 均匀正则防止坍塌。"""
        gamma = float(self.gamma_alpha if gamma_alpha is None else gamma_alpha)
        eps = float(self.epsilon if epsilon is None else epsilon)
        eps = min(max(eps, 0.0), 1.0)

        with torch.no_grad():
            u = utilities.detach().to(self.score_device, dtype=torch.float32)
            alpha_tilde = self.alpha * torch.exp(gamma * u)
            alpha_tilde = alpha_tilde.clamp_min(1e-20)
            alpha_norm = alpha_tilde / alpha_tilde.sum().clamp_min(1e-12)
            self.alpha = (1.0 - eps) * alpha_norm + (eps / float(self.k))
            self.alpha = self.alpha / self.alpha.sum().clamp_min(1e-12)
            return self.alpha.detach().clone()

    def train_on_sampled_batch(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        gradient_accumulation_steps: int = 1,
        loss_divisor: float = 1.0,
        zero_grad: bool = True,
        step_optimizer: bool = True,
    ) -> float:
        """用采样 batch 做一次参数更新（支持梯度累积）。

        参数：
        - gradient_accumulation_steps: 把一个 batch 切成多个微批次累积梯度，
          最后只执行一次 optimizer.step()。
        - loss_divisor: 额外 loss 缩放因子（用于跨采样批次的梯度累积）。
        - zero_grad: 是否在本次调用前清梯度。
        - step_optimizer: 是否在本次调用后执行 optimizer.step()。
        """
        self.model.train()

        dyn_keys, dyn_vals = self._split_dynamic_batch_fields(batch)
        if not dyn_keys:
            raise ValueError("No dynamic tensor fields found in batch for training.")
        bsz = int(dyn_vals[0].size(0))
        if bsz <= 0:
            raise ValueError("Empty batch is not allowed for training.")

        accum_steps = max(1, int(gradient_accumulation_steps))
        micro_bsz = max(1, (bsz + accum_steps - 1) // accum_steps)

        micro_ranges: List[Tuple[int, int]] = []
        for start in range(0, bsz, micro_bsz):
            end = min(bsz, start + micro_bsz)
            if end > start:
                micro_ranges.append((start, end))
        num_micro = max(1, len(micro_ranges))

        if zero_grad:
            self.optimizer.zero_grad(set_to_none=True)

        loss_sum = 0.0
        for start, end in micro_ranges:
            micro_batch: Dict[str, torch.Tensor] = {}
            for key, value in batch.items():
                if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == bsz:
                    micro_batch[key] = value[start:end]
                else:
                    micro_batch[key] = value

            outputs = self.model(**micro_batch)
            loss = self.loss_fn(outputs, micro_batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss encountered: {loss.item()}")

            total_div = float(num_micro) * max(float(loss_divisor), 1e-12)
            (loss / total_div).backward()
            loss_sum += float(loss.detach().item())

        if step_optimizer:
            self.optimizer.step()
        return loss_sum / float(num_micro)

    # -----------------------------
    # 单步执行：1->2->3->4
    # -----------------------------
    def step(
        self,
        *,
        batch_size: int,
        anchor_size_per_capability: int,
        anchor_indices_by_capability: Optional[Mapping[int, torch.Tensor]] = None,
        anchor_chunk_size: int = 8,
        replacement: bool = False,
        gamma_T: Optional[float] = None,
        eta_beta: Optional[float] = None,
        gamma_alpha: Optional[float] = None,
        epsilon: Optional[float] = None,
    ) -> StepStats:
        """执行一次完整 Online Recipe Evolution。"""
        # 阶段 1：锚点梯度
        if anchor_indices_by_capability is None:
            anchor_indices_by_capability = self.sample_anchor_indices(anchor_size_per_capability)
        current_anchor_grads = self.compute_anchor_gradients(
            anchor_indices_by_capability=anchor_indices_by_capability,
            chunk_size=anchor_chunk_size,
        )
        anchor_grads = self.update_running_anchor_gradients(current_anchor_grads)

        # 阶段 2：beta 更新
        beta_new = self.update_beta(anchor_grads, eta_beta=eta_beta)

        # 阶段 3：全局打分 + 采样
        sampled_idx, _probs, _scores = self.score_and_sample_batch(
            batch_size=batch_size,
            gamma_T=gamma_T,
            replacement=replacement,
        )
        sampled_batch = self._gather_pool_batch(sampled_idx)

        # 阶段 4：单样本梯度 -> 奖励 -> alpha 更新
        per_sample_grads = self.compute_per_sample_layer_gradients_vmap(sampled_batch)
        rewards = self.compute_local_rewards(
            batch_indices=sampled_idx,
            per_sample_grads=per_sample_grads,
            anchor_gradients=anchor_grads,
        )
        self.update_pruning_state(batch_indices=sampled_idx, rewards=rewards)

        # 先完成模型训练，再更新 alpha（alpha/beta 与模型图解耦）。
        train_loss = self.train_on_sampled_batch(sampled_batch)
        utilities = self.compute_mapper_utilities(batch_indices=sampled_idx, rewards=rewards)
        alpha_new = self.update_alpha(utilities, gamma_alpha=gamma_alpha, epsilon=epsilon)

        self.step_id += 1
        self.alpha_history.append(alpha_new.detach().clone().cpu())
        self.beta_history.append(beta_new.detach().clone().cpu())

        return StepStats(
            step=self.step_id,
            train_loss=train_loss,
            avg_reward=float(rewards.mean().item()) if rewards.numel() > 0 else 0.0,
            sampled_indices=sampled_idx.detach().clone(),
            alpha=alpha_new.detach().clone(),
            beta=beta_new.detach().clone(),
            utilities=utilities.detach().clone(),
        )


__all__ = ["RecipeEvolver", "StepStats"]
