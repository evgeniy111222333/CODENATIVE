from __future__ import annotations

import torch
import torch.nn.functional as F


def autoregressive_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[0] < 2:
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits[:-1], targets[:-1])


def masked_autoregressive_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    supervision_mask: torch.Tensor | None,
) -> torch.Tensor:
    if supervision_mask is None or logits.shape[0] < 2:
        return logits.new_tensor(0.0)
    effective_mask = supervision_mask[:-1]
    if not bool(effective_mask.any().item()):
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits[:-1][effective_mask], targets[:-1][effective_mask])


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


def symbol_link_loss(
    candidate_scores: list[torch.Tensor],
    candidate_node_ids: list[tuple[str, ...]],
    target_node_ids: list[str | None],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for scores, node_ids, target_node_id in zip(candidate_scores, candidate_node_ids, target_node_ids, strict=False):
        if target_node_id is None or scores.numel() == 0 or target_node_id not in node_ids:
            continue
        target_index = node_ids.index(target_node_id)
        losses.append(
            F.cross_entropy(
                scores.unsqueeze(0),
                torch.tensor([target_index], dtype=torch.long, device=scores.device),
            )
        )
    if not losses:
        device = candidate_scores[0].device if candidate_scores else None
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def definition_use_loss(
    candidate_scores: list[torch.Tensor],
    candidate_node_ids: list[tuple[str, ...]],
    candidate_kinds: list[tuple[str, ...]],
    target_node_ids: list[str | None],
) -> torch.Tensor:
    return _filtered_candidate_ce_loss(
        candidate_scores=candidate_scores,
        candidate_node_ids=candidate_node_ids,
        candidate_kinds=candidate_kinds,
        target_node_ids=target_node_ids,
        required_kinds={"symbol", "function", "class"},
    )


def diagnostic_probe_loss(
    candidate_scores: list[torch.Tensor],
    candidate_node_ids: list[tuple[str, ...]],
    candidate_kinds: list[tuple[str, ...]],
    target_node_ids: list[str | None],
) -> torch.Tensor:
    return _filtered_candidate_ce_loss(
        candidate_scores=candidate_scores,
        candidate_node_ids=candidate_node_ids,
        candidate_kinds=candidate_kinds,
        target_node_ids=target_node_ids,
        required_kinds={"diagnostic", "test", "config"},
    )


def routing_loss(
    router_post_logits: list[torch.Tensor],
    teacher_indices: list[int],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for logits, teacher_index in zip(router_post_logits, teacher_indices, strict=False):
        losses.append(
            F.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([teacher_index], dtype=torch.long, device=logits.device),
            )
        )
    if not losses:
        device = router_post_logits[0].device if router_post_logits else None
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def route_consistency_loss(
    router_pre_logits: list[torch.Tensor],
    teacher_expensive: list[tuple[int, int, int]],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for logits, teacher in zip(router_pre_logits, teacher_expensive, strict=False):
        target = torch.tensor(teacher, dtype=torch.float32, device=logits.device)
        losses.append(F.binary_cross_entropy_with_logits(logits, target))
    if not losses:
        device = router_pre_logits[0].device if router_pre_logits else None
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def router_oracle_loss(
    router_post_logits: list[torch.Tensor],
    oracle_weights: list[torch.Tensor],
    router_post_masks: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    masks = router_post_masks or [None] * len(router_post_logits)
    for logits, oracle, mask in zip(router_post_logits, oracle_weights, masks, strict=False):
        if oracle.numel() == 0 or not bool((oracle > 0).any().item()):
            continue
        masked_logits = logits
        if mask is not None:
            masked_logits = logits.clone()
            masked_logits[~mask] = -1e9
        log_probs = F.log_softmax(masked_logits, dim=-1)
        losses.append(F.kl_div(log_probs, oracle, reduction="batchmean"))
    if not losses:
        device = router_post_logits[0].device if router_post_logits else None
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def router_entropy_floor_loss(
    router_weights: torch.Tensor | list[torch.Tensor] | None,
    min_entropy: float,
) -> torch.Tensor:
    if router_weights is None:
        return torch.tensor(0.0)
    if isinstance(router_weights, list):
        if not router_weights:
            return torch.tensor(0.0)
        weights = torch.stack(router_weights, dim=0)
    else:
        weights = router_weights
    if weights.numel() == 0:
        return weights.new_tensor(0.0)
    normalized = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    entropies = (-normalized * torch.log(normalized.clamp_min(1e-8))).sum(dim=-1)
    return torch.relu(entropies.new_tensor(min_entropy) - entropies).mean()


def energy_penalty(energy_proxy: torch.Tensor | None, always_on_energy: float) -> torch.Tensor:
    if energy_proxy is None or energy_proxy.numel() == 0:
        device = energy_proxy.device if isinstance(energy_proxy, torch.Tensor) else None
        return torch.tensor(0.0, device=device)
    baseline = energy_proxy.new_tensor(always_on_energy)
    return torch.relu(energy_proxy - baseline).mean()


def cap_auxiliary_losses(
    ar_loss: torch.Tensor,
    auxiliary_losses: dict[str, torch.Tensor],
    max_ratio: float,
) -> dict[str, torch.Tensor]:
    capped: dict[str, torch.Tensor] = {}
    cap_value = ar_loss.detach() * max_ratio
    for name, loss in auxiliary_losses.items():
        if loss.detach().item() <= cap_value.item():
            capped[name] = loss
            continue
        scale = cap_value / loss.detach().clamp_min(1e-8)
        capped[name] = loss * scale
    return capped


def _filtered_candidate_ce_loss(
    candidate_scores: list[torch.Tensor],
    candidate_node_ids: list[tuple[str, ...]],
    candidate_kinds: list[tuple[str, ...]],
    target_node_ids: list[str | None],
    required_kinds: set[str],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for scores, node_ids, kinds, target_node_id in zip(
        candidate_scores,
        candidate_node_ids,
        candidate_kinds,
        target_node_ids,
        strict=False,
    ):
        if target_node_id is None or scores.numel() == 0 or target_node_id not in node_ids:
            continue
        if not required_kinds.intersection(kinds):
            continue
        target_index = node_ids.index(target_node_id)
        losses.append(
            F.cross_entropy(
                scores.unsqueeze(0),
                torch.tensor([target_index], dtype=torch.long, device=scores.device),
            )
        )
    if not losses:
        device = candidate_scores[0].device if candidate_scores else None
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()
