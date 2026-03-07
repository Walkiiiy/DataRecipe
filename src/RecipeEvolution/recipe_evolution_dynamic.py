"""Single-objective RecipeEvolution with gradient-alignment feedback.

Implements:
- DataSelector: dynamic alpha over 4 static metrics, top-k selection.
- Trainer: anchor gradient + per-sample gradient rewards via torch.func.vmap/grad.
- RecipeEvolution: orchestration of selection -> training -> alpha update.

Includes a runnable dummy demo in ``main()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap


@dataclass
class BatchData:
    """Mini-batch container.

    Shapes:
        x: model input batch.
        y: labels. For classification: ``(batch_size,)``.
           For token tasks: ``(batch_size, seq_len)``.
        metrics: ``(batch_size, 4)``
        loss_mask: Optional target mask aligned with ``y``.
           1 means contribute to loss, 0 means masked-out.
    """

    x: torch.Tensor
    y: torch.Tensor
    metrics: torch.Tensor
    loss_mask: Optional[torch.Tensor] = None


class DataSelector:
    """Maintain alpha and perform score-based top-k selection.

    Args:
        num_metrics: Number of static metrics (fixed to 4 in this setup).
        device: Torch device.
    """

    def __init__(self, num_metrics: int = 4, device: torch.device | None = None) -> None:
        if num_metrics != 4:
            raise ValueError("This simplified setup expects num_metrics=4.")
        self.device = device if device is not None else torch.device("cpu")
        self.alpha = torch.full((num_metrics,), 1.0 / num_metrics, device=self.device)

    def score_pool(self, metrics: torch.Tensor) -> torch.Tensor:
        """Compute scalar score for each sample.

        Args:
            metrics: Metric matrix with shape ``(num_samples, 4)``.

        Returns:
            Score vector with shape ``(num_samples,)``.
        """
        return metrics @ self.alpha

    def select_topk(self, x: torch.Tensor, y: torch.Tensor, metrics: torch.Tensor, k: int) -> BatchData:
        """Select top-k data by current alpha-weighted score."""
        scores = self.score_pool(metrics)
        topk_idx = torch.topk(scores, k=min(k, scores.numel()), dim=0, largest=True).indices
        return BatchData(x=x[topk_idx], y=y[topk_idx], metrics=metrics[topk_idx])

    def update_alpha(
        self,
        rewards: torch.Tensor,
        batch_metrics: torch.Tensor,
        gamma: float,
        epsilon: float,
    ) -> torch.Tensor:
        """Batch-smoothed MWU update.

        U_i = E_d[R(d) * M_i(d)]
        alpha_tilde_i = alpha_i * exp(gamma * U_i)
        alpha_i = (1-epsilon) * alpha_tilde_i/sum(alpha_tilde) + epsilon/4

        Args:
            rewards: Reward vector ``(batch_size,)``.
            batch_metrics: Metric matrix ``(batch_size, 4)``.
            gamma: MWU learning rate.
            epsilon: Entropy regularization factor.

        Returns:
            Updated alpha ``(4,)``.
        """
        utilities = (rewards.unsqueeze(1) * batch_metrics).mean(dim=0)
        alpha_tilde = self.alpha * torch.exp(gamma * utilities)
        alpha_norm = alpha_tilde / alpha_tilde.sum().clamp_min(1e-12)
        self.alpha = (1.0 - epsilon) * alpha_norm + epsilon / 4.0
        return self.alpha


class SimpleLinearLM(nn.Module):
    """Tiny model for demo.

    Forward input/output shapes:
        x: ``(batch_size, input_dim)``
        logits: ``(batch_size, num_classes)``
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.lm_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.lm_head(h)


class Trainer:
    """Train model and compute gradient-alignment rewards.

    Args:
        model: Trainable model with ``lm_head`` as final layer.
        optimizer: Torch optimizer.
        target_val_set: Tuple ``(x_val, y_val)`` with shapes
            ``(val_size, input_dim)``, ``(val_size,)``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        target_val_set: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.x_val, self.y_val = target_val_set

    @staticmethod
    def _masked_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute CE loss with optional output mask.

        Supported:
        - classification logits ``(B, C)``, labels ``(B,)``
        - token logits ``(B, T, C)``, labels ``(B, T)``
        """
        if logits.dim() == 2 and labels.dim() == 1:
            # Classification: each sample has a single target.
            return F.cross_entropy(logits, labels)

        if logits.dim() == 3 and labels.dim() == 2:
            # Token task: explicit masking over non-target positions.
            if loss_mask is None:
                loss_mask = torch.ones_like(labels, dtype=torch.bool)
            else:
                loss_mask = loss_mask.to(dtype=torch.bool)

            masked_labels = labels.masked_fill(~loss_mask, -100)
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                masked_labels.reshape(-1),
                ignore_index=-100,
            )

        raise ValueError(
            f"Unsupported shapes for masked loss: logits={tuple(logits.shape)}, "
            f"labels={tuple(labels.shape)}"
        )

    def _flatten_last_layer_grad_from_param_dict(self, grad_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        w = grad_dict["lm_head.weight"].reshape(-1)
        if "lm_head.bias" in grad_dict:
            b = grad_dict["lm_head.bias"].reshape(-1)
            return torch.cat([w, b], dim=0)
        return w

    def _anchor_gradient(self) -> torch.Tensor:
        """Compute g_ref from validation set without optimizer update."""
        self.model.zero_grad(set_to_none=True)

        logits = self.model(self.x_val)
        loss = self._masked_loss(logits, self.y_val, loss_mask=None)
        loss.backward()

        grads = {
            "lm_head.weight": self.model.lm_head.weight.grad.detach().clone(),
        }
        if self.model.lm_head.bias is not None:
            grads["lm_head.bias"] = self.model.lm_head.bias.grad.detach().clone()

        g_ref = self._flatten_last_layer_grad_from_param_dict(grads)
        g_ref = g_ref / g_ref.norm().clamp_min(1e-12)

        self.model.zero_grad(set_to_none=True)
        return g_ref

    def _per_sample_last_layer_grads(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-sample last-layer gradients with torch.func.vmap/grad.

        Returns:
            Tensor with shape ``(batch_size, last_layer_num_params)``.
        """
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())

        def single_loss(
            p: Dict[str, torch.Tensor],
            b: Dict[str, torch.Tensor],
            x_i: torch.Tensor,
            y_i: torch.Tensor,
            mask_i: Optional[torch.Tensor],
        ) -> torch.Tensor:
            logits_i = functional_call(self.model, (p, b), (x_i.unsqueeze(0),))
            if y_i.dim() == 0:
                labels_i = y_i.unsqueeze(0)
                mask_i_b = None
            else:
                labels_i = y_i.unsqueeze(0)
                mask_i_b = None if mask_i is None else mask_i.unsqueeze(0)
            return self._masked_loss(logits_i, labels_i, loss_mask=mask_i_b)

        grad_fn = grad(single_loss)
        if loss_mask is None:
            per_sample = vmap(grad_fn, in_dims=(None, None, 0, 0, None))(
                params,
                buffers,
                x,
                y,
                None,
            )
        else:
            per_sample = vmap(grad_fn, in_dims=(None, None, 0, 0, 0))(
                params,
                buffers,
                x,
                y,
                loss_mask,
            )

        # Extract final-layer per-sample gradients.
        g_w = per_sample["lm_head.weight"].reshape(x.size(0), -1)
        if "lm_head.bias" in per_sample:
            g_b = per_sample["lm_head.bias"].reshape(x.size(0), -1)
            g = torch.cat([g_w, g_b], dim=1)
        else:
            g = g_w

        g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return g

    def train_step(self, batch: BatchData) -> Tuple[torch.Tensor, float]:
        """Perform one training step and return rewards + train loss."""
        g_ref = self._anchor_gradient()
        g_d = self._per_sample_last_layer_grads(batch.x, batch.y, batch.loss_mask)

        # Reward: cosine(g_d, g_ref)
        rewards = g_d @ g_ref

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(batch.x)
        loss = self._masked_loss(logits, batch.y, batch.loss_mask)
        loss.backward()
        self.optimizer.step()

        return rewards.detach(), float(loss.item())


class RecipeEvolution:
    """End-to-end controller: select -> train -> update alpha."""

    def __init__(
        self,
        selector: DataSelector,
        trainer: Trainer,
        x_pool: torch.Tensor,
        y_pool: torch.Tensor,
        m_pool: torch.Tensor,
        gamma: float = 0.8,
        epsilon: float = 0.05,
    ) -> None:
        self.selector = selector
        self.trainer = trainer
        self.x_pool = x_pool
        self.y_pool = y_pool
        self.m_pool = m_pool
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, batch_size: int) -> Dict[str, torch.Tensor | float]:
        batch = self.selector.select_topk(self.x_pool, self.y_pool, self.m_pool, k=batch_size)
        rewards, train_loss = self.trainer.train_step(batch)
        alpha = self.selector.update_alpha(
            rewards=rewards,
            batch_metrics=batch.metrics,
            gamma=self.gamma,
            epsilon=self.epsilon,
        )
        return {
            "train_loss": train_loss,
            "avg_reward": float(rewards.mean().item()),
            "alpha": alpha.detach().clone(),
        }


def main() -> None:
    torch.manual_seed(7)

    device = torch.device("cpu")

    # Dummy candidate pool
    num_samples = 256
    input_dim = 32
    hidden_dim = 48
    num_classes = 7

    x_pool = torch.randn(num_samples, input_dim, device=device)
    y_pool = torch.randint(0, num_classes, (num_samples,), device=device)
    m_pool = torch.rand(num_samples, 4, device=device)  # static metric matrix M^(d)

    # Dummy target validation set (anchor set)
    x_val = torch.randn(64, input_dim, device=device)
    y_val = torch.randint(0, num_classes, (64,), device=device)

    model = SimpleLinearLM(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    selector = DataSelector(num_metrics=4, device=device)
    trainer = Trainer(model=model, optimizer=optimizer, target_val_set=(x_val, y_val))

    evolution = RecipeEvolution(
        selector=selector,
        trainer=trainer,
        x_pool=x_pool,
        y_pool=y_pool,
        m_pool=m_pool,
        gamma=0.8,
        epsilon=0.05,
    )

    print("Initial alpha:", selector.alpha.tolist())
    for epoch in range(1, 6):
        stats = evolution.step(batch_size=32)
        print(
            f"epoch={epoch} loss={stats['train_loss']:.4f} "
            f"avg_reward={stats['avg_reward']:.4f} alpha={stats['alpha'].tolist()}"
        )


if __name__ == "__main__":
    main()
