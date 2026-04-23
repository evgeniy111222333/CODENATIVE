from __future__ import annotations

from collections import defaultdict

import torch

from htm_code_native.config.settings import HTMCodeNativeConfig
from htm_code_native.data.types import TrainingPhase


GROUP_SPECS = {
    "backbone": {"lr_scale": 1.0, "clip_norm": 1.0},
    "semantic_memory": {"lr_scale": 0.7, "clip_norm": 0.5},
    "erm": {"lr_scale": 0.7, "clip_norm": 0.5},
    "eem": {"lr_scale": 0.5, "clip_norm": 0.5},
    "router_heads": {"lr_scale": 0.7, "clip_norm": 0.25},
}


def build_optimizer(
    model: torch.nn.Module,
    config: HTMCodeNativeConfig,
    phase: TrainingPhase,
    warmup_active: bool,
) -> torch.optim.Optimizer:
    base_lr = config.model.optimizer_base_lr
    router_scale = 0.3 if warmup_active else 0.7
    grouped_params: dict[str, list[torch.nn.Parameter]] = defaultdict(list)

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        grouped_params[_group_name_for_parameter(name)].append(parameter)

    parameter_groups = []
    for group_name in ("backbone", "semantic_memory", "erm", "eem", "router_heads"):
        params = grouped_params.get(group_name, [])
        if not params:
            continue
        lr_scale = router_scale if group_name == "router_heads" else GROUP_SPECS[group_name]["lr_scale"]
        parameter_groups.append(
            {
                "params": params,
                "lr": base_lr * lr_scale,
                "group_name": group_name,
                "clip_norm": GROUP_SPECS[group_name]["clip_norm"],
                "phase": phase.value,
            }
        )
    return torch.optim.AdamW(parameter_groups, lr=base_lr)


def clip_grad_groups(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    gradient_norms: dict[str, float] = {}
    for param_group in optimizer.param_groups:
        params = [parameter for parameter in param_group["params"] if parameter.grad is not None]
        if not params:
            continue
        clip_norm = float(param_group.get("clip_norm", 1.0))
        group_name = str(param_group.get("group_name", "group"))
        gradient_norms[group_name] = float(
            torch.nn.utils.clip_grad_norm_(params, max_norm=clip_norm).item()
        )
    return gradient_norms


def _group_name_for_parameter(name: str) -> str:
    if name.startswith(
        (
            "encoder.",
            "hssm.",
            "master_norm.",
            "level_output_projections.",
            "skip_projection.",
            "semantic_projection.",
            "graph_out_projection.",
            "graph_query_projection.",
            "hidden_ffn.",
        )
    ) or name == "level_gate_vectors":
        return "backbone"
    if name.startswith("semantic_memory."):
        return "semantic_memory"
    if name.startswith("exact_recent_memory."):
        return "erm"
    if name.startswith("exact_episodic_memory."):
        return "eem"
    return "router_heads"
