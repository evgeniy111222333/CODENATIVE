from __future__ import annotations

import torch

from htm_code_native.config.settings import HSSMConfig, SemanticMemoryConfig
from htm_code_native.data.types import HSSMState
from htm_code_native.hssm.core import HSSMCore
from htm_code_native.memory.semantic.store import SemanticMemory


def test_hssm_respects_update_masks() -> None:
    config = HSSMConfig(max_level=2, hidden_size=8, stride_base=2, norm_clip=10.0)
    module = HSSMCore(config)
    embeddings = torch.randn(4, 8)
    boundaries = {
        0: torch.tensor([True, True, True, True]),
        1: torch.tensor([True, False, False, False]),
        2: torch.tensor([True, False, False, False]),
    }
    output = module(embeddings, boundaries)
    assert bool(output.update_mask[1, 1].item()) is False
    assert bool(output.update_mask[2, 1].item()) is True


def test_state_norm_projection_clips_drift() -> None:
    config = HSSMConfig(max_level=1, hidden_size=8, stride_base=2, norm_clip=0.5)
    module = HSSMCore(config)
    embeddings = torch.full((3, 8), 50.0)
    boundaries = {
        0: torch.tensor([True, True, True]),
        1: torch.tensor([True, True, True]),
    }
    output = module(embeddings, boundaries)
    assert torch.linalg.norm(output.level_states, dim=-1).max().item() <= 0.5001


def test_semantic_memory_hot_read_and_consolidation() -> None:
    config = HSSMConfig(max_level=1, hidden_size=8, stride_base=2, norm_clip=10.0)
    memory = SemanticMemory(
        hidden_size=8,
        hssm_config=config,
        config=SemanticMemoryConfig(
            key_dim=8,
            hot_slots=4,
            cold_slots=8,
            beam_width=2,
            consolidation_fill_threshold=0.5,
            maintenance_budget=1.0,
            min_slots_for_consolidation=2,
        ),
    )
    memory.reset()
    for step in range(4):
        base = torch.ones(8) * (step + 1)
        state = HSSMState(
            level_states=[base, base],
            last_update_indices=[step, step],
            master_state=torch.cat([base, base]),
            step_index=step,
        )
        result = memory.read_write(state, budget=1.0)
    assert result.hot_reads >= 0
    assert memory.cold_clusters[0] or memory.cold_clusters[1]
