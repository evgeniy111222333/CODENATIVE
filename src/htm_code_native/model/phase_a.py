from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from htm_code_native.config.settings import HTMCodeNativeConfig
from htm_code_native.data.types import HSSMState, PhaseABatch, PhaseAOutput, RepoGraphQueryContext, RepositoryGraphIndex
from htm_code_native.encoders.code import CodeAwareEmbedding
from htm_code_native.memory.exact_episodic import ExactEpisodicMemory
from htm_code_native.hssm.core import HSSMCore
from htm_code_native.memory.exact_recent import ExactRecentMemory
from htm_code_native.memory.repo_graph import RepositoryGraphMemory
from htm_code_native.memory.semantic.store import SemanticMemory


class PhaseACodeModel(nn.Module):
    def __init__(self, config: HTMCodeNativeConfig) -> None:
        super().__init__()
        if config.model.model_dim != config.hssm.hidden_size:
            raise ValueError("Phase A expects model.model_dim == hssm.hidden_size.")
        if (
            config.model.semantic_blend
            + config.model.erm_blend
            + config.model.eem_blend
            + config.model.graph_blend
            > 1.0
        ):
            raise ValueError("semantic_blend + erm_blend + eem_blend + graph_blend must not exceed 1.0.")

        self.config = config
        hidden_size = config.model.model_dim
        num_levels = config.hssm.num_levels
        master_dim = hidden_size * num_levels
        self.repo_graph_root: Path | None = None

        self.encoder = CodeAwareEmbedding(config)
        self.hssm = HSSMCore(config.hssm)
        self.semantic_memory = SemanticMemory(hidden_size, config.hssm, config.semantic_memory)
        self.exact_recent_memory = ExactRecentMemory(
            hidden_size=hidden_size,
            key_dim=config.model.erm_key_dim,
            window_size=config.model.recent_window,
            vocab_size=config.model.vocabulary_size,
            max_byte_payload=config.model.max_recent_byte_payload,
        )
        self.exact_episodic_memory = ExactEpisodicMemory(
            hidden_size=hidden_size,
            key_dim=config.model.eem_key_dim,
            pointer_key_dim=config.model.pointer_key_dim,
            vocab_size=config.model.vocabulary_size,
            top_k=config.model.eem_top_k,
            max_chunk_tokens=config.model.max_chunk_tokens,
            max_chunks=config.model.max_episodic_chunks,
        )
        self.repo_graph_memory = RepositoryGraphMemory(
            hidden_size=hidden_size,
            key_dim=config.model.graph_key_dim,
            vocab_size=config.model.vocabulary_size,
            top_k=config.model.graph_top_k,
            graph_copy_weight=config.model.graph_copy_weight,
            samefile_bias=config.model.graph_samefile_bias,
            import_bias=config.model.graph_import_bias,
            symbol_bias=config.model.graph_symbol_bias,
            test_bias=config.model.graph_test_bias,
            diagnostic_bias=config.model.graph_diagnostic_bias,
        )

        self.master_norm = nn.LayerNorm(master_dim)
        self.level_gate_vectors = nn.Parameter(torch.randn(num_levels, master_dim))
        self.level_output_projections = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_levels)]
        )
        self.skip_projection = nn.Linear(hidden_size, hidden_size)
        self.semantic_projection = nn.Linear(hidden_size, hidden_size)
        self.hidden_ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.vocab_head = nn.Linear(hidden_size, config.model.vocabulary_size)
        self.semantic_head = (
            nn.Linear(hidden_size, config.model.vocabulary_size)
            if config.model.use_semantic_head
            else None
        )

    def set_repo_graph_index(self, index: RepositoryGraphIndex | None) -> None:
        self.repo_graph_memory.set_index(index)
        self.repo_graph_root = Path(index.root_path).resolve() if index is not None else None

    def forward(self, batch: PhaseABatch, reset_eem: bool = True) -> PhaseAOutput:
        embeddings, encoder_parts = self.encoder(batch)
        hssm_output = self.hssm(embeddings, batch.boundaries)
        self.semantic_memory.reset()
        self.exact_recent_memory.reset()
        self.repo_graph_memory.reset()
        if reset_eem:
            self.exact_episodic_memory.reset()

        seq_len = embeddings.shape[0]
        num_levels = self.config.hssm.num_levels
        hidden_size = self.config.model.model_dim

        hidden_states = torch.zeros(seq_len, hidden_size, device=embeddings.device)
        semantic_contexts = torch.zeros(seq_len, hidden_size, device=embeddings.device)
        logits = torch.zeros(
            seq_len,
            self.config.model.vocabulary_size,
            device=embeddings.device,
        )
        lm_logits = torch.zeros_like(logits)
        semantic_logits = (
            torch.zeros_like(logits) if self.semantic_head is not None else None
        )
        erm_logits = torch.full_like(logits, fill_value=math.log(1e-8))
        erm_attention = torch.zeros(
            seq_len,
            self.config.model.recent_window,
            device=embeddings.device,
        )
        copy_target_mask = torch.zeros(seq_len, dtype=torch.bool, device=embeddings.device)
        eem_logits = torch.full_like(logits, fill_value=math.log(1e-8))
        eem_attention = torch.zeros(
            seq_len,
            self.config.model.eem_top_k,
            device=embeddings.device,
        )
        pointer_attention = torch.zeros(
            seq_len,
            self.config.model.eem_top_k * self.config.model.max_chunk_tokens,
            device=embeddings.device,
        )
        episodic_target_mask = torch.zeros(seq_len, dtype=torch.bool, device=embeddings.device)
        graph_logits = torch.full_like(logits, fill_value=math.log(1e-8))
        graph_attention = torch.zeros(
            seq_len,
            self.config.model.graph_top_k,
            device=embeddings.device,
        )
        graph_copy_target_mask = torch.zeros(seq_len, dtype=torch.bool, device=embeddings.device)
        entropy_tensor = torch.zeros(seq_len, num_levels, device=embeddings.device)
        total_hot_reads = 0
        total_cold_reads = 0
        total_maintenance = 0
        total_erm_reads = 0
        total_erm_writes = 0
        total_erm_overwrites = 0
        copy_target_hits = 0
        total_eem_reads = 0
        total_chunks_finalized = 0
        total_chunk_overhead = 0.0
        episodic_target_hits = 0
        total_graph_reads = 0
        total_graph_candidates = 0
        total_graph_samefile_hits = 0
        total_graph_import_hits = 0
        total_graph_symbol_hits = 0
        total_graph_test_hits = 0
        total_graph_diagnostic_hits = 0
        graph_copy_hits = 0
        current_chunk_start = 0
        graph_candidate_ids: list[tuple[str, ...]] = []
        graph_candidate_kinds: list[tuple[str, ...]] = []
        graph_candidate_names: list[tuple[str, ...]] = []

        for step in range(seq_len):
            step_level_states = [
                hssm_output.level_states[step, level] for level in range(num_levels)
            ]
            step_state = HSSMState(
                level_states=step_level_states,
                last_update_indices=hssm_output.last_update_indices,
                master_state=hssm_output.master_states[step],
                step_index=step,
            )
            read_result = self.semantic_memory.read_write(
                step_state,
                budget=self.config.semantic_memory.maintenance_budget,
            )
            total_hot_reads += read_result.hot_reads
            total_cold_reads += read_result.cold_reads
            total_maintenance += read_result.maintenance_invocations

            gamma = torch.softmax(
                (self.level_gate_vectors @ self.master_norm(step_state.master_state))
                / math.sqrt(self.config.model.model_dim),
                dim=0,
            )
            stacked_outputs = torch.stack(read_result.per_level_outputs, dim=0)
            projected_outputs = torch.stack(
                [
                    self.level_output_projections[level](stacked_outputs[level])
                    for level in range(num_levels)
                ],
                dim=0,
            )
            semantic_context = torch.sum(gamma.unsqueeze(-1) * projected_outputs, dim=0)
            hidden = self.hidden_ffn(
                self.semantic_projection(semantic_context)
                + self.skip_projection(hssm_output.level_states[step, 0])
            )
            step_lm_logits = self.vocab_head(hidden)
            lm_probabilities = torch.softmax(step_lm_logits, dim=-1)
            semantic_probabilities = None
            if self.semantic_head is not None:
                step_semantic_logits = self.semantic_head(semantic_context)
                semantic_logits[step] = step_semantic_logits
                semantic_probabilities = torch.softmax(step_semantic_logits, dim=-1)
            else:
                step_semantic_logits = None

            payload_length = int(batch.token_payload_lengths[step].item())
            token_span = batch.token_spans[step].tolist()
            payload = batch.document.token_bytes(step)[:payload_length]
            overwrite = self.exact_recent_memory.write(
                step_state=hssm_output.level_states[step, 0],
                token_id=int(batch.token_ids[step].item()),
                span=(int(token_span[0]), int(token_span[1])),
                payload=payload,
                timestamp=step,
            )
            exact_recent_result = self.exact_recent_memory.read(hidden)
            erm_logits[step] = exact_recent_result.log_distribution
            erm_attention[step] = exact_recent_result.attention
            total_erm_reads += exact_recent_result.read_count
            total_erm_writes += 1
            total_erm_overwrites += int(overwrite)

            if step < seq_len - 1:
                target_token_id = int(batch.targets[step].item())
                has_copy_target = bool((exact_recent_result.slot_token_ids == target_token_id).any().item())
                copy_target_mask[step] = has_copy_target
                copy_target_hits += int(has_copy_target)

            finalized_chunk = None
            chunk_type = self._chunk_type_for_step(batch, step, current_chunk_start)
            if chunk_type is not None:
                finalized_chunk = self.exact_episodic_memory.maybe_finalize_chunk(
                    document=batch.document,
                    batch=batch,
                    level0_states=hssm_output.level_states[:, 0],
                    start_index=current_chunk_start,
                    end_index=step,
                    chunk_type=chunk_type,
                    timestamp=step,
                )
                if finalized_chunk is not None:
                    total_chunks_finalized += 1
                    current_chunk_start = step + 1

            exact_episodic_result = self.exact_episodic_memory.retrieve(hidden)
            eem_logits[step] = exact_episodic_result.log_distribution
            eem_attention[step] = exact_episodic_result.chunk_attention
            pointer_attention[step] = exact_episodic_result.pointer_attention
            total_eem_reads += exact_episodic_result.read_count
            total_chunk_overhead += exact_episodic_result.chunk_overhead

            if step < seq_len - 1:
                target_token_id = int(batch.targets[step].item())
                has_episodic_target = bool(
                    (exact_episodic_result.pointer_token_ids == target_token_id).any().item()
                )
                episodic_target_mask[step] = has_episodic_target
                episodic_target_hits += int(has_episodic_target)

            structure = batch.document.token_structures[step]
            graph_context = RepoGraphQueryContext(
                file_path=self._normalize_graph_path(structure.file_id),
                current_symbol_id=structure.symbol_id,
                current_symbol_name=structure.symbol_name,
                scope_path=structure.scope_path,
                token_value=batch.document.tokens[step].value,
                token_class=batch.document.tokens[step].token_class.value,
            )
            graph_result = self.repo_graph_memory.query(
                hidden,
                graph_context,
                batch.vocabulary_snapshot,
            )
            graph_logits[step] = graph_result.log_distribution
            graph_attention[step] = graph_result.attention
            graph_candidate_ids.append(graph_result.candidate_node_ids)
            graph_candidate_kinds.append(graph_result.candidate_kinds)
            graph_candidate_names.append(graph_result.candidate_names)
            total_graph_reads += graph_result.read_count
            total_graph_candidates += graph_result.candidate_count
            total_graph_samefile_hits += graph_result.samefile_hits
            total_graph_import_hits += graph_result.import_hits
            total_graph_symbol_hits += graph_result.symbol_hits
            total_graph_test_hits += graph_result.test_hits
            total_graph_diagnostic_hits += graph_result.diagnostic_hits

            if step < seq_len - 1 and graph_result.copy_token_ids.numel() > 0:
                target_token_id = int(batch.targets[step].item())
                has_graph_copy = bool((graph_result.copy_token_ids == target_token_id).any().item())
                graph_copy_target_mask[step] = has_graph_copy
                graph_copy_hits += int(has_graph_copy)

            semantic_weight = (
                self.config.model.semantic_blend if semantic_probabilities is not None else 0.0
            )
            erm_weight = (
                self.config.model.erm_blend if exact_recent_result.filled_size > 0 else 0.0
            )
            eem_weight = (
                self.config.model.eem_blend if exact_episodic_result.retrieved_chunk_count > 0 else 0.0
            )
            graph_weight = (
                self.config.model.graph_blend if graph_result.retrieved_count > 0 else 0.0
            )
            lm_weight = 1.0 - semantic_weight - erm_weight - eem_weight - graph_weight
            if lm_weight < 0.0:
                lm_weight = 0.0
            total_weight = lm_weight + semantic_weight + erm_weight + eem_weight + graph_weight
            if total_weight == 0.0:
                total_weight = 1.0
                lm_weight = 1.0

            blended_probabilities = (lm_weight / total_weight) * lm_probabilities
            if semantic_probabilities is not None:
                blended_probabilities = blended_probabilities + (
                    semantic_weight / total_weight
                ) * semantic_probabilities
            if exact_recent_result.filled_size > 0:
                blended_probabilities = blended_probabilities + (
                    erm_weight / total_weight
                ) * exact_recent_result.distribution
            if exact_episodic_result.retrieved_chunk_count > 0:
                blended_probabilities = blended_probabilities + (
                    eem_weight / total_weight
                ) * exact_episodic_result.distribution
            if graph_result.retrieved_count > 0:
                blended_probabilities = blended_probabilities + (
                    graph_weight / total_weight
                ) * graph_result.distribution
            blended = torch.log(blended_probabilities.clamp_min(1e-8))

            lm_logits[step] = step_lm_logits
            hidden_states[step] = hidden
            semantic_contexts[step] = semantic_context
            logits[step] = blended
            for level, entropy in read_result.entropies.items():
                entropy_tensor[step, level] = entropy

        diagnostics = {
            "mean_update_rate": float(hssm_output.update_mask.float().mean().item()),
            "mean_entropy": float(entropy_tensor.mean().item()),
            "embedding_norm": float(embeddings.norm(dim=-1).mean().item()),
        }
        memory_stats = {
            "hot_reads": float(total_hot_reads),
            "cold_reads": float(total_cold_reads),
            "maintenance_invocations": float(total_maintenance),
            "erm_reads": float(total_erm_reads),
            "erm_writes": float(total_erm_writes),
            "erm_fill": float(self.exact_recent_memory.filled),
            "erm_overwrites": float(total_erm_overwrites),
            "copy_target_hits": float(copy_target_hits),
            "eem_reads": float(total_eem_reads),
            "chunks_finalized": float(total_chunks_finalized),
            "stored_chunks": float(len(self.exact_episodic_memory.chunks)),
            "avg_chunk_overhead": float(total_chunk_overhead / max(seq_len, 1)),
            "episodic_target_hits": float(episodic_target_hits),
            "graph_reads": float(total_graph_reads),
            "graph_candidates": float(total_graph_candidates),
            "graph_copy_hits": float(graph_copy_hits),
            "graph_samefile_hits": float(total_graph_samefile_hits),
            "graph_import_hits": float(total_graph_import_hits),
            "graph_symbol_hits": float(total_graph_symbol_hits),
            "graph_test_hits": float(total_graph_test_hits),
            "graph_diagnostic_hits": float(total_graph_diagnostic_hits),
        }

        return PhaseAOutput(
            logits=logits,
            lm_logits=lm_logits,
            semantic_logits=semantic_logits,
            erm_logits=erm_logits,
            erm_attention=erm_attention,
            copy_target_mask=copy_target_mask,
            eem_logits=eem_logits,
            eem_attention=eem_attention,
            pointer_attention=pointer_attention,
            episodic_target_mask=episodic_target_mask,
            graph_logits=graph_logits,
            graph_attention=graph_attention,
            graph_copy_target_mask=graph_copy_target_mask,
            hidden_states=hidden_states,
            semantic_contexts=semantic_contexts,
            diagnostics=diagnostics,
            memory_stats=memory_stats,
            auxiliary={
                "encoder_parts": encoder_parts,
                "level_states": hssm_output.level_states,
                "master_states": hssm_output.master_states,
                "update_mask": hssm_output.update_mask,
                "lower_aggregates": hssm_output.lower_aggregates,
                "entropy_tensor": entropy_tensor,
                "vocabulary_snapshot": batch.vocabulary_snapshot,
                "graph_candidate_ids": graph_candidate_ids,
                "graph_candidate_kinds": graph_candidate_kinds,
                "graph_candidate_names": graph_candidate_names,
            },
        )

    def _chunk_type_for_step(self, batch: PhaseABatch, step: int, current_chunk_start: int) -> str | None:
        chunk_length = step - current_chunk_start + 1
        if chunk_length <= 0:
            return None
        if bool(batch.boundaries[3][step].item()):
            return "callable"
        if bool(batch.boundaries[2][step].item()) and chunk_length >= self.config.model.min_chunk_tokens:
            return "block"
        if bool(batch.boundaries[4][step].item()):
            return "file"
        if chunk_length >= self.config.model.max_chunk_tokens:
            return "threshold"
        return None

    def _normalize_graph_path(self, file_path: str) -> str:
        path = Path(file_path)
        if self.repo_graph_root is None:
            return path.as_posix()
        try:
            return path.resolve().relative_to(self.repo_graph_root).as_posix()
        except (ValueError, RuntimeError):
            return path.as_posix()
