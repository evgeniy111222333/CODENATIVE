from __future__ import annotations

from htm_code_native.config.settings import HTMCodeNativeConfig
from htm_code_native.data.types import MaintenanceDecision, TrainingPhase


def update_ar_ema(previous_ema: float | None, ar_loss: float, decay: float) -> float:
    if previous_ema is None:
        return ar_loss
    return (decay * previous_ema) + ((1.0 - decay) * ar_loss)


def schedule_maintenance(
    step_index: int,
    metrics: dict[str, float],
    config: HTMCodeNativeConfig,
    phase: TrainingPhase,
) -> MaintenanceDecision:
    hot_occupancy = float(metrics.get("hot_occupancy", 0.0))
    ar_loss = float(metrics.get("ar_loss", 0.0))
    ar_ema = float(metrics.get("ar_ema", ar_loss))
    ar_delta = abs(ar_loss - ar_ema)
    under_warmup = step_index <= config.model.router_warmup_steps
    cadence = max(config.model.maintenance_cadence, 1)
    cadence_hit = step_index > 0 and (step_index % cadence == 0)

    if phase == TrainingPhase.PHASE_A:
        return MaintenanceDecision(
            should_consolidate=False,
            hot_occupancy=hot_occupancy,
            ar_ema=ar_ema,
            ar_delta=ar_delta,
            cadence_hit=cadence_hit,
            under_warmup=under_warmup,
            reason="phase_a_disabled",
        )
    if under_warmup:
        return MaintenanceDecision(
            should_consolidate=False,
            hot_occupancy=hot_occupancy,
            ar_ema=ar_ema,
            ar_delta=ar_delta,
            cadence_hit=cadence_hit,
            under_warmup=True,
            reason="under_warmup",
        )
    if hot_occupancy < config.semantic_memory.consolidation_fill_threshold:
        return MaintenanceDecision(
            should_consolidate=False,
            hot_occupancy=hot_occupancy,
            ar_ema=ar_ema,
            ar_delta=ar_delta,
            cadence_hit=cadence_hit,
            under_warmup=False,
            reason="low_hot_occupancy",
        )
    if not cadence_hit:
        return MaintenanceDecision(
            should_consolidate=False,
            hot_occupancy=hot_occupancy,
            ar_ema=ar_ema,
            ar_delta=ar_delta,
            cadence_hit=False,
            under_warmup=False,
            reason="cadence_miss",
        )
    if ar_delta > config.model.maintenance_loss_spike_delta:
        return MaintenanceDecision(
            should_consolidate=False,
            hot_occupancy=hot_occupancy,
            ar_ema=ar_ema,
            ar_delta=ar_delta,
            cadence_hit=True,
            under_warmup=False,
            reason="ar_spike_guard",
        )
    return MaintenanceDecision(
        should_consolidate=True,
        hot_occupancy=hot_occupancy,
        ar_ema=ar_ema,
        ar_delta=ar_delta,
        cadence_hit=True,
        under_warmup=False,
        reason="scheduled",
    )
