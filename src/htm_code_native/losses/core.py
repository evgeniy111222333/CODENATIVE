from __future__ import annotations

import torch
import torch.nn.functional as F


def autoregressive_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[0] < 2:
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits[:-1], targets[:-1])


def hierarchical_consistency_loss(
    level_states: torch.Tensor,
    lower_aggregates: torch.Tensor,
    update_mask: torch.Tensor,
) -> torch.Tensor:
    if level_states.shape[1] <= 1:
        return level_states.new_tensor(0.0)

    total = level_states.new_tensor(0.0)
    count = 0
    for level in range(1, level_states.shape[1]):
        mask = update_mask[:, level]
        if mask.any():
            diff = level_states[:, level][mask] - lower_aggregates[:, level][mask]
            total = total + diff.pow(2).mean()
            count += 1
    if count == 0:
        return total
    return total / count


def sparse_retrieval_entropy_loss(entropy_tensor: torch.Tensor) -> torch.Tensor:
    if entropy_tensor.numel() == 0:
        return entropy_tensor.new_tensor(0.0)
    return entropy_tensor.mean()


def recent_copy_loss(
    erm_logits: torch.Tensor | None,
    targets: torch.Tensor,
    copy_target_mask: torch.Tensor | None,
) -> torch.Tensor:
    if erm_logits is None or copy_target_mask is None or not bool(copy_target_mask.any().item()):
        device = targets.device if isinstance(targets, torch.Tensor) else None
        return torch.tensor(0.0, device=device)
    return F.nll_loss(erm_logits[copy_target_mask], targets[copy_target_mask])


def episodic_pointer_loss(
    eem_logits: torch.Tensor | None,
    targets: torch.Tensor,
    episodic_target_mask: torch.Tensor | None,
) -> torch.Tensor:
    if eem_logits is None or episodic_target_mask is None or not bool(episodic_target_mask.any().item()):
        device = targets.device if isinstance(targets, torch.Tensor) else None
        return torch.tensor(0.0, device=device)
    return F.nll_loss(eem_logits[episodic_target_mask], targets[episodic_target_mask])


def graph_copy_loss(
    graph_logits: torch.Tensor | None,
    targets: torch.Tensor,
    graph_copy_target_mask: torch.Tensor | None,
) -> torch.Tensor:
    if graph_logits is None or graph_copy_target_mask is None or not bool(graph_copy_target_mask.any().item()):
        device = targets.device if isinstance(targets, torch.Tensor) else None
        return torch.tensor(0.0, device=device)
    return F.nll_loss(graph_logits[graph_copy_target_mask], targets[graph_copy_target_mask])
