from __future__ import annotations

from htm_code_native.data.types import TrainingPhase
from htm_code_native.model.phase_a import PhaseACodeModel
from htm_code_native.training.maintenance import schedule_maintenance
from htm_code_native.training.optimizer import build_optimizer


def test_optimizer_groups_and_router_warmup_lr(config) -> None:
    model = PhaseACodeModel(config)
    optimizer_warm = build_optimizer(model, config, TrainingPhase.PHASE_D, warmup_active=True)
    warm_groups = {group["group_name"]: group for group in optimizer_warm.param_groups}
    assert {"backbone", "semantic_memory", "erm", "eem", "router_heads"} <= set(warm_groups)
    assert warm_groups["backbone"]["clip_norm"] == 1.0
    assert warm_groups["semantic_memory"]["clip_norm"] == 0.5
    assert warm_groups["router_heads"]["clip_norm"] == 0.25
    assert warm_groups["router_heads"]["lr"] == config.model.optimizer_base_lr * 0.3

    optimizer_stable = build_optimizer(model, config, TrainingPhase.PHASE_E, warmup_active=False)
    stable_groups = {group["group_name"]: group for group in optimizer_stable.param_groups}
    assert stable_groups["router_heads"]["lr"] == config.model.optimizer_base_lr * 0.7


def test_maintenance_scheduler_blocks_warmup_and_loss_spike(config) -> None:
    config.model.semantic_maintenance_warmup_steps = 8
    warmup_decision = schedule_maintenance(
        1,
        {"hot_occupancy": 1.0, "ar_loss": 0.5, "ar_ema": 0.5},
        config,
        TrainingPhase.PHASE_D,
    )
    assert warmup_decision.should_consolidate is False
    assert warmup_decision.reason == "under_warmup"

    config.model.semantic_maintenance_warmup_steps = 0
    early_decision = schedule_maintenance(
        1,
        {"hot_occupancy": 1.0, "ar_loss": 0.5, "ar_ema": 0.5},
        config,
        TrainingPhase.PHASE_D,
    )
    assert early_decision.reason == "cadence_miss"

    step_index = config.model.maintenance_cadence
    spike_decision = schedule_maintenance(
        step_index,
        {
            "hot_occupancy": 1.0,
            "ar_loss": 1.0,
            "ar_ema": 1.0 - (config.model.maintenance_loss_spike_delta * 2.0),
        },
        config,
        TrainingPhase.PHASE_D,
    )
    assert spike_decision.should_consolidate is False
    assert spike_decision.reason == "ar_spike_guard"

    scheduled_decision = schedule_maintenance(
        step_index,
        {"hot_occupancy": 1.0, "ar_loss": 0.5, "ar_ema": 0.5},
        config,
        TrainingPhase.PHASE_D,
    )
    assert scheduled_decision.should_consolidate is True
    assert scheduled_decision.reason == "scheduled"
