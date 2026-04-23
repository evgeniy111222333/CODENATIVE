from __future__ import annotations

import torch

from htm_code_native.data.types import TrainingPhase
from htm_code_native.losses.core import router_entropy_floor_loss
from htm_code_native.router.stub import TwoStageRouter


def build_router(lane_dropout_prob: float = 0.0, collapse_window: int = 32, threshold: float = 0.95) -> TwoStageRouter:
    return TwoStageRouter(
        pre_feature_dim=8,
        post_feature_dim=10,
        pre_hidden_dim=16,
        post_hidden_dim=16,
        temperature=1.0,
        route_top_k=1,
        thresholds=(0.35, 0.35, 0.35),
        lane_costs=(1.0, 1.0, 1.0, 2.0, 2.5, 2.5),
        warmup_steps=8,
        oracle_sharpness=4.0,
        oracle_biases=(0.0, 0.0, 0.0, 0.0, 0.0),
        lane_dropout_prob=lane_dropout_prob,
        collapse_mass_threshold=threshold,
        collapse_window=collapse_window,
        recovery_steps=4,
    )


def test_tilde_rho_interpolates_across_beta() -> None:
    router = build_router()
    learned = torch.tensor([0.1, 0.2, 0.7, 0.0, 0.0])
    post_mask = torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool)
    oracle = torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool)

    oracle_weights, effective_weights, beta, warmup_active, _, _, _, _, _ = router.apply_warmup(
        post_logits=torch.zeros(5),
        learned_weights=learned,
        post_mask=post_mask,
        oracle_availability=oracle,
        phase=TrainingPhase.PHASE_B,
        global_step=0,
        training=True,
    )
    assert warmup_active is True
    assert beta == 0.0
    assert torch.allclose(effective_weights, oracle_weights)

    _, effective_weights, beta, warmup_active, _, _, _, _, _ = router.apply_warmup(
        post_logits=torch.zeros(5),
        learned_weights=learned,
        post_mask=post_mask,
        oracle_availability=oracle,
        phase=TrainingPhase.PHASE_B,
        global_step=router.warmup_steps,
        training=True,
    )
    assert warmup_active is False
    assert beta == 1.0
    assert torch.allclose(effective_weights, learned)


def test_dominant_lane_dropout_preserves_distribution() -> None:
    torch.manual_seed(0)
    router = build_router(lane_dropout_prob=1.0)
    learned = torch.tensor([0.92, 0.04, 0.04, 0.0, 0.0])
    post_mask = torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool)
    oracle = torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool)

    _, effective_weights, _, warmup_active, dominant_lane_dropped, _, _, _, _ = router.apply_warmup(
        post_logits=torch.zeros(5),
        learned_weights=learned,
        post_mask=post_mask,
        oracle_availability=oracle,
        phase=TrainingPhase.PHASE_B,
        global_step=0,
        training=True,
    )
    assert warmup_active is True
    assert dominant_lane_dropped is True
    assert torch.isclose(effective_weights.sum(), torch.tensor(1.0))
    assert float(effective_weights.max().item()) < 1.0


def test_entropy_floor_loss_behaves_as_expected() -> None:
    peaked = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    flat = torch.full((1, 5), 0.2)
    assert float(router_entropy_floor_loss(peaked, min_entropy=1.1).item()) > 0.0
    assert float(router_entropy_floor_loss(flat, min_entropy=1.1).item()) == 0.0


def test_collapse_detector_triggers_after_window() -> None:
    router = build_router(collapse_window=3, threshold=0.8)
    learned = torch.tensor([0.99, 0.01, 0.0, 0.0, 0.0])
    post_mask = torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)
    oracle = torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)

    collapse = False
    for step in range(3):
        _, _, _, _, _, collapse, _, _, remaining = router.apply_warmup(
            post_logits=torch.zeros(5),
            learned_weights=learned,
            post_mask=post_mask,
            oracle_availability=oracle,
            phase=TrainingPhase.PHASE_B,
            global_step=step,
            training=True,
        )
    assert collapse is True
    assert remaining >= 0


def test_router_runtime_state_resumes_collapse_history() -> None:
    router = build_router(collapse_window=3, threshold=0.8)
    learned = torch.tensor([0.99, 0.01, 0.0, 0.0, 0.0])
    post_mask = torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)
    oracle = torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)

    for step in range(2):
        router.apply_warmup(
            post_logits=torch.zeros(5),
            learned_weights=learned,
            post_mask=post_mask,
            oracle_availability=oracle,
            phase=TrainingPhase.PHASE_B,
            global_step=step,
            training=True,
        )
    resumed = build_router(collapse_window=3, threshold=0.8)
    resumed.load_state(router.export_state())
    fresh = build_router(collapse_window=3, threshold=0.8)

    _, _, _, _, _, resumed_collapse, _, _, resumed_remaining = resumed.apply_warmup(
        post_logits=torch.zeros(5),
        learned_weights=learned,
        post_mask=post_mask,
        oracle_availability=oracle,
        phase=TrainingPhase.PHASE_B,
        global_step=2,
        training=True,
    )
    _, _, _, _, _, fresh_collapse, _, _, fresh_remaining = fresh.apply_warmup(
        post_logits=torch.zeros(5),
        learned_weights=learned,
        post_mask=post_mask,
        oracle_availability=oracle,
        phase=TrainingPhase.PHASE_B,
        global_step=2,
        training=True,
    )

    assert resumed_collapse is True
    assert fresh_collapse is False
    assert resumed_remaining >= 0
    assert fresh_remaining == 0
