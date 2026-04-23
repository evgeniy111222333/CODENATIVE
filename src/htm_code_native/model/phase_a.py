from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from htm_code_native.config.settings import HTMCodeNativeConfig
from htm_code_native.data.types import (
    ExactEpisodicReadResult,
    ExactRecentReadResult,
    HSSMState,
    PhaseABatch,
    PhaseAOutput,
    RepoGraphQueryContext,
    RepoGraphReadResult,
    RepositoryGraphIndex,
    RouterFeatures,
    TaskLabel,
    TokenClass,
    TrainingPhase,
)
from htm_code_native.encoders.code import CodeAwareEmbedding
from htm_code_native.hssm.core import HSSMCore
from htm_code_native.memory.exact_episodic import ExactEpisodicMemory
from htm_code_native.memory.exact_recent import ExactRecentMemory
from htm_code_native.memory.repo_graph import RepositoryGraphMemory
from htm_code_native.memory.semantic.store import SemanticMemory
from htm_code_native.router.stub import TwoStageRouter


class PhaseACodeModel(nn.Module):
    def __init__(self, config: HTMCodeNativeConfig) -> None:
        super().__init__()
        if config.model.model_dim != config.hssm.hidden_size:
            raise ValueError("Phase A expects model.model_dim == hssm.hidden_size.")

        self.config = config
        hidden_size = config.model.model_dim
        num_levels = config.hssm.num_levels
        master_dim = hidden_size * num_levels
        self.repo_graph_root: Path | None = None
        self.router_metadata_dim = 21

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
            value_dim=config.model.graph_value_dim,
        )
        self.router = TwoStageRouter(
            pre_feature_dim=master_dim + hidden_size + self.router_metadata_dim,
            post_feature_dim=master_dim + hidden_size + self.router_metadata_dim + 5,
            pre_hidden_dim=config.model.pre_router_hidden_dim,
            post_hidden_dim=config.model.post_router_hidden_dim,
            temperature=config.model.route_temperature,
            route_top_k=config.model.route_top_k,
            thresholds=(
                config.model.route_threshold_cold_semantic,
                config.model.route_threshold_eem,
                config.model.route_threshold_graph,
            ),
            lane_costs=(
                config.model.lane_cost_lm,
                config.model.lane_cost_semantic_hot,
                config.model.lane_cost_erm,
                config.model.lane_cost_semantic_cold,
                config.model.lane_cost_eem,
                config.model.lane_cost_graph,
            ),
            warmup_steps=config.model.router_warmup_steps,
            oracle_sharpness=config.model.router_oracle_sharpness,
            oracle_biases=(
                config.model.router_oracle_bias_lm,
                config.model.router_oracle_bias_semantic,
                config.model.router_oracle_bias_erm,
                config.model.router_oracle_bias_eem,
                config.model.router_oracle_bias_graph,
            ),
            lane_dropout_prob=config.model.router_lane_dropout_prob,
            collapse_mass_threshold=config.model.router_collapse_mass_threshold,
            collapse_window=config.model.router_collapse_window,
            recovery_steps=config.model.router_recovery_steps,
        )

        self.master_norm = nn.LayerNorm(master_dim)
        self.level_gate_vectors = nn.Parameter(torch.randn(num_levels, master_dim))
        self.level_output_projections = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_levels)]
        )
        self.skip_projection = nn.Linear(hidden_size, hidden_size)
        self.semantic_projection = nn.Linear(hidden_size, hidden_size)
        self.graph_out_projection = nn.Linear(config.model.graph_value_dim, hidden_size)
        self.graph_query_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.hidden_ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.edit_span_head = nn.Linear(hidden_size + config.model.graph_value_dim, 1)
        self.vocab_head = nn.Linear(hidden_size, config.model.vocabulary_size)
        self.semantic_head = (
            nn.Linear(hidden_size, config.model.vocabulary_size)
            if config.model.use_semantic_head
            else None
        )

    def set_repo_graph_index(self, index: RepositoryGraphIndex | None) -> None:
        self.repo_graph_memory.set_index(index)
        self.repo_graph_root = Path(index.root_path).resolve() if index is not None else None

    def forward(
        self,
        batch: PhaseABatch,
        reset_eem: bool = True,
        phase: TrainingPhase | str | None = None,
        task_label: TaskLabel | str | None = None,
        global_step: int = 0,
        maintenance_budget: float = 0.0,
    ) -> PhaseAOutput:
        phase_name = self._resolve_phase(phase)
        task_name = self._resolve_task_label(batch, task_label)
        embeddings, encoder_parts = self.encoder(batch)
        hssm_output = self.hssm(embeddings, batch.boundaries)
        self.semantic_memory.reset()
        self.exact_recent_memory.reset()
        self.repo_graph_memory.reset()
        self.router.reset()
        if reset_eem:
            self.exact_episodic_memory.reset()

        seq_len = embeddings.shape[0]
        num_levels = self.config.hssm.num_levels
        hidden_size = self.config.model.model_dim
        device = embeddings.device
        token_counts = Counter(token.value if token.value else token.token_type for token in batch.document.tokens)

        base_hidden_states = torch.zeros(seq_len, hidden_size, device=device)
        hidden_states = torch.zeros(seq_len, hidden_size, device=device)
        semantic_contexts = torch.zeros(seq_len, hidden_size, device=device)
        graph_contexts = torch.zeros(seq_len, self.config.model.graph_value_dim, device=device)
        logits = torch.zeros(seq_len, self.config.model.vocabulary_size, device=device)
        lm_logits = torch.zeros_like(logits)
        semantic_logits = torch.zeros_like(logits) if self.semantic_head is not None else None
        erm_logits = torch.full_like(logits, fill_value=math.log(1e-8))
        eem_logits = torch.full_like(logits, fill_value=math.log(1e-8))
        graph_logits = torch.full_like(logits, fill_value=math.log(1e-8))
        erm_attention = torch.zeros(seq_len, self.config.model.recent_window, device=device)
        eem_attention = torch.zeros(seq_len, self.config.model.eem_top_k, device=device)
        pointer_attention = torch.zeros(
            seq_len,
            self.config.model.eem_top_k * self.config.model.max_chunk_tokens,
            device=device,
        )
        graph_attention = torch.zeros(seq_len, self.config.model.graph_top_k, device=device)
        copy_target_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        episodic_target_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        graph_copy_target_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        entropy_tensor = torch.zeros(seq_len, num_levels, device=device)
        lane_entropies = torch.zeros(seq_len, 5, device=device)
        router_weights = torch.zeros(seq_len, 5, device=device)
        effective_router_weights = torch.zeros(seq_len, 5, device=device)
        oracle_router_weights = torch.zeros(seq_len, 5, device=device)
        oracle_availability = torch.zeros(seq_len, 5, dtype=torch.bool, device=device)
        router_pre_mask = torch.zeros(seq_len, 6, dtype=torch.bool, device=device)
        router_post_mask = torch.zeros(seq_len, 5, dtype=torch.bool, device=device)
        invoked_lanes = torch.zeros(seq_len, 6, dtype=torch.bool, device=device)
        energy_proxy = torch.zeros(seq_len, device=device)
        warmup_beta = torch.ones(seq_len, device=device)
        collapse_detected = torch.zeros(seq_len, dtype=torch.bool, device=device)
        edit_token_scores = torch.zeros(seq_len, device=device)

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
        graph_fusion_steps = 0
        graph_task_eligible_steps = 0
        cold_semantic_invocations = 0
        eem_invocations = 0
        graph_invocations = 0
        symbol_link_hits = 0
        warmup_active_steps = 0
        dominant_lane_drop_steps = 0
        router_collapse_steps = 0
        current_chunk_start = 0
        previous_lane_stats = torch.zeros(5, device=device)
        graph_candidate_ids: list[tuple[str, ...]] = []
        graph_candidate_kinds: list[tuple[str, ...]] = []
        graph_candidate_names: list[tuple[str, ...]] = []
        graph_candidate_scores: list[torch.Tensor] = []
        graph_target_node_ids: list[str | None] = []
        route_teacher_indices: list[int] = []
        route_teacher_expensive: list[tuple[int, int, int]] = []
        router_pre_logits: list[torch.Tensor] = []
        router_post_logits: list[torch.Tensor] = []
        router_post_masks: list[torch.Tensor] = []
        oracle_router_weight_list: list[torch.Tensor] = []
        warmup_steps_remaining = 0
        phase_policy = self._phase_policy(phase_name)

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

            hot_outputs, hot_entropies, hot_reads = self.semantic_memory.read_hot(step_state)
            hot_context = self._aggregate_semantic_context(step_state.master_state, hot_outputs)
            base_hidden = self.hidden_ffn(
                self.semantic_projection(hot_context)
                + self.skip_projection(hssm_output.level_states[step, 0])
            )

            cold_available = phase_policy["cold_semantic_enabled"] and any(
                self.semantic_memory.cold_clusters[level] for level in range(num_levels)
            )
            erm_enabled = phase_policy["erm_enabled"]
            eem_enabled = phase_policy["eem_enabled"]
            if not eem_enabled:
                current_chunk_start = step + 1
            eem_available = eem_enabled and bool(self.exact_episodic_memory.chunks)
            graph_available = bool(self.repo_graph_memory.index and self.repo_graph_memory.index.nodes)
            graph_task_relevant = graph_available and self._graph_task_is_relevant(batch, step)
            graph_enabled = phase_policy["graph_enabled"] and graph_task_relevant
            if phase_name == TrainingPhase.PHASE_D:
                graph_enabled = graph_enabled and task_name in {TaskLabel.REPO_GRAPH, TaskLabel.EDIT_FIX}
            graph_task_eligible_steps += int(graph_task_relevant)
            metadata = self._router_metadata(
                batch=batch,
                step=step,
                token_counts=token_counts,
                previous_lane_stats=previous_lane_stats,
                availability=(cold_available, eem_available, graph_enabled),
            )
            pre_features = torch.cat(
                [self.master_norm(step_state.master_state), base_hidden, metadata],
                dim=-1,
            )

            cold_context = torch.zeros(hidden_size, device=device)
            cold_entropies: dict[int, float] = {level: 0.0 for level in range(num_levels)}
            cold_reads = 0

            availability_mask = torch.tensor(
                [cold_available, eem_available, graph_enabled],
                dtype=torch.bool,
                device=device,
            )
            pre_router_features = RouterFeatures(
                pre_features=pre_features,
                post_features=torch.zeros(
                    self.master_norm(step_state.master_state).shape[0] + hidden_size + self.router_metadata_dim + 5,
                    device=device,
                ),
                availability_mask=availability_mask,
                phase=phase_name,
                task_label=task_name,
                step_index=global_step,
                always_on_pre_mask=phase_policy["always_on_pre_mask"].to(device),
                allowed_post_mask=phase_policy["allowed_post_mask"].to(device),
            )
            warmup_flag = self.training and phase_policy["warmup_enabled"] and global_step < self.config.model.router_warmup_steps
            pre_logits, _expensive_probs, pre_mask, base_energy_proxy, _always_on_energy = self.router.route_pre(
                pre_router_features,
                warmup_active=warmup_flag,
            )

            if bool(pre_mask[3].item()) and cold_available:
                cold_outputs, cold_entropies, cold_reads = self.semantic_memory.read_cold(step_state)
                cold_context = self._aggregate_semantic_context(step_state.master_state, cold_outputs)
                cold_semantic_invocations += 1
            else:
                cold_outputs = [torch.zeros(hidden_size, device=device) for _ in range(num_levels)]

            graph_query_hidden = self.graph_query_projection(
                torch.cat([base_hidden, encoder_parts["struct_component"][step]], dim=-1)
            )
            if bool(pre_mask[5].item()) and graph_enabled:
                graph_result = self.repo_graph_memory.query(
                    hidden=graph_query_hidden,
                    context=self._graph_query_context(batch, step),
                    vocabulary_snapshot=batch.vocabulary_snapshot,
                )
                graph_invocations += 1
                graph_fusion_steps += int(graph_result.retrieved_count > 0)
            else:
                graph_result = self._empty_graph_result(device)

            graph_context = (
                graph_result.graph_context
                if graph_result.retrieved_count > 0
                else torch.zeros(self.config.model.graph_value_dim, device=device)
            )
            semantic_context = hot_context + cold_context + (
                self.config.model.graph_out_blend * self.graph_out_projection(graph_context)
            )
            hidden = self.hidden_ffn(
                self.semantic_projection(semantic_context)
                + self.skip_projection(hssm_output.level_states[step, 0])
            )

            step_lm_logits = self.vocab_head(hidden)
            lm_probabilities = torch.softmax(step_lm_logits, dim=-1)
            if self.semantic_head is not None:
                step_semantic_logits = self.semantic_head(semantic_context)
                semantic_probabilities = torch.softmax(step_semantic_logits, dim=-1)
                semantic_logits[step] = step_semantic_logits
            else:
                step_semantic_logits = None
                semantic_probabilities = None

            if erm_enabled:
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
                total_erm_reads += exact_recent_result.read_count
                total_erm_writes += 1
                total_erm_overwrites += int(overwrite)
            else:
                exact_recent_result = self._empty_erm_result(device)
            erm_logits[step] = exact_recent_result.log_distribution
            erm_attention[step] = exact_recent_result.attention

            if eem_enabled:
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

            if bool(pre_mask[4].item()) and eem_available:
                exact_episodic_result = self.exact_episodic_memory.retrieve(hidden)
                if eem_enabled:
                    eem_invocations += 1
            else:
                exact_episodic_result = self._empty_eem_result(device)
            eem_logits[step] = exact_episodic_result.log_distribution
            eem_attention[step] = exact_episodic_result.chunk_attention
            pointer_attention[step] = exact_episodic_result.pointer_attention
            total_eem_reads += exact_episodic_result.read_count
            total_chunk_overhead += exact_episodic_result.chunk_overhead

            graph_logits[step] = graph_result.log_distribution
            graph_attention[step] = graph_result.attention
            graph_candidate_ids.append(graph_result.candidate_node_ids)
            graph_candidate_kinds.append(graph_result.candidate_kinds)
            graph_candidate_names.append(graph_result.candidate_names)
            graph_candidate_scores.append(graph_result.candidate_scores)
            total_graph_reads += graph_result.read_count
            total_graph_candidates += graph_result.candidate_count
            total_graph_samefile_hits += graph_result.samefile_hits
            total_graph_import_hits += graph_result.import_hits
            total_graph_symbol_hits += graph_result.symbol_hits
            total_graph_test_hits += graph_result.test_hits
            total_graph_diagnostic_hits += graph_result.diagnostic_hits

            semantic_entropy_value = self._distribution_entropy(
                semantic_probabilities if semantic_probabilities is not None else lm_probabilities
            )
            erm_entropy_value = self._distribution_entropy(exact_recent_result.distribution)
            eem_entropy_value = self._distribution_entropy(exact_episodic_result.distribution)
            graph_entropy_value = self._distribution_entropy(graph_result.distribution)
            lm_entropy_value = self._distribution_entropy(lm_probabilities)
            lane_entropy = torch.tensor(
                [
                    lm_entropy_value,
                    semantic_entropy_value,
                    erm_entropy_value,
                    eem_entropy_value,
                    graph_entropy_value,
                ],
                dtype=torch.float32,
                device=device,
            )
            lane_entropies[step] = lane_entropy

            post_features = torch.cat(
                [
                    self.master_norm(step_state.master_state),
                    hidden,
                    metadata,
                    lane_entropy,
                ],
                dim=-1,
            )
            post_router_features = RouterFeatures(
                pre_features=pre_features,
                post_features=post_features,
                availability_mask=availability_mask,
                phase=phase_name,
                task_label=task_name,
                step_index=global_step,
                always_on_pre_mask=phase_policy["always_on_pre_mask"].to(device),
                allowed_post_mask=phase_policy["allowed_post_mask"].to(device),
            )

            for level in range(num_levels):
                entropy_tensor[step, level] = hot_entropies.get(level, 0.0) + cold_entropies.get(level, 0.0)

            has_copy_target = False
            has_episodic_target = False
            graph_target_node_id = None
            if step < seq_len - 1:
                target_token_id = int(batch.targets[step].item())
                has_copy_target = bool((exact_recent_result.slot_token_ids == target_token_id).any().item())
                copy_target_mask[step] = has_copy_target
                copy_target_hits += int(has_copy_target)

                has_episodic_target = bool(
                    (exact_episodic_result.pointer_token_ids == target_token_id).any().item()
                )
                episodic_target_mask[step] = has_episodic_target
                episodic_target_hits += int(has_episodic_target)

                next_structure = batch.document.token_structures[step + 1]
                has_graph_copy = bool((graph_result.copy_token_ids == target_token_id).any().item())
                graph_copy_target_mask[step] = has_graph_copy
                graph_copy_hits += int(has_graph_copy)
                graph_target_node_id = self._resolve_graph_target_node(
                    next_structure.symbol_id,
                    next_structure.symbol_name,
                    graph_result.candidate_node_ids,
                    graph_result.candidate_names,
                )
                symbol_link_hits += int(graph_target_node_id is not None)
                graph_target_node_ids.append(graph_target_node_id)

                teacher_index, teacher_expensive = self._route_teacher(
                    batch=batch,
                    step=step,
                    copy_hit=has_copy_target,
                    episodic_hit=has_episodic_target,
                    graph_hit=bool(graph_copy_target_mask[step].item()) or graph_target_node_id is not None,
                    phase=phase_name,
                )
                route_teacher_indices.append(teacher_index)
                route_teacher_expensive.append(teacher_expensive)
            else:
                graph_target_node_ids.append(None)
                route_teacher_indices.append(0)
                route_teacher_expensive.append((0, 0, 0))
                graph_copy_target_mask[step] = False

            post_logits, learned_weights, post_mask = self.router.route_post(
                post_router_features,
                pre_mask=pre_mask,
            )
            if task_name == TaskLabel.EDIT_FIX:
                if eem_available:
                    post_mask[3] = True
                if graph_enabled:
                    post_mask[4] = True
                masked_logits = post_logits.clone()
                masked_logits[~post_mask] = -1e9
                learned_weights = torch.softmax(masked_logits / self.config.model.route_temperature, dim=-1)
                learned_weights = torch.where(post_mask, learned_weights, torch.zeros_like(learned_weights))
                learned_weights = learned_weights / learned_weights.sum().clamp_min(1e-8)
            oracle_mask = self._oracle_availability(
                copy_hit=erm_enabled and has_copy_target,
                episodic_hit=eem_enabled and has_episodic_target,
                graph_hit=graph_enabled and (bool(graph_copy_target_mask[step].item()) or graph_target_node_id is not None),
                post_mask=post_mask,
                phase=phase_name,
            )
            oracle_availability[step] = oracle_mask
            (
                oracle_weights,
                effective_weights,
                step_warmup_beta,
                step_warmup_active,
                dominant_lane_dropped,
                step_collapse_detected,
                router_entropy,
                dominant_lane_mass,
                warmup_steps_remaining,
            ) = self.router.apply_warmup(
                post_logits=post_logits,
                learned_weights=learned_weights,
                post_mask=post_mask,
                oracle_availability=oracle_mask,
                phase=phase_name,
                global_step=global_step,
                training=self.training,
            )
            router_pre_logits.append(pre_logits)
            router_post_logits.append(post_logits)
            router_post_masks.append(post_mask.clone())
            oracle_router_weight_list.append(oracle_weights.clone())
            router_pre_mask[step] = pre_mask
            invoked_lanes[step] = pre_mask
            router_post_mask[step] = post_mask
            router_weights[step] = learned_weights
            if task_name == TaskLabel.EDIT_FIX:
                edit_prior = torch.tensor([1.0, 1.0, 1.0, 1.15, 1.25], device=device)
                effective_weights = effective_weights * edit_prior
                effective_weights = effective_weights / effective_weights.sum().clamp_min(1e-8)
            effective_router_weights[step] = effective_weights
            oracle_router_weights[step] = oracle_weights
            warmup_beta[step] = step_warmup_beta
            collapse_detected[step] = step_collapse_detected
            warmup_active_steps += int(step_warmup_active)
            dominant_lane_drop_steps += int(dominant_lane_dropped)
            router_collapse_steps += int(step_collapse_detected)

            blended_probabilities = (
                effective_weights[0] * lm_probabilities
                + effective_weights[1]
                * (
                    semantic_probabilities
                    if semantic_probabilities is not None
                    else lm_probabilities
                )
                + effective_weights[2] * exact_recent_result.distribution
                + effective_weights[3] * exact_episodic_result.distribution
                + effective_weights[4] * graph_result.distribution
            )
            logits[step] = torch.log(blended_probabilities.clamp_min(1e-8))

            lm_logits[step] = step_lm_logits
            base_hidden_states[step] = base_hidden
            hidden_states[step] = hidden
            semantic_contexts[step] = semantic_context
            graph_contexts[step] = graph_context
            edit_token_scores[step] = self.edit_span_head(torch.cat([hidden, graph_context], dim=-1)).squeeze(-1)

            self.semantic_memory.write_hot(step_state)
            if maintenance_budget > 0.0:
                with torch.no_grad():
                    step_maintenance = self.semantic_memory.consolidate(
                        maintenance_budget,
                        step_state.step_index,
                    )
            else:
                step_maintenance = 0
            total_maintenance += step_maintenance
            total_hot_reads += hot_reads
            total_cold_reads += cold_reads
            energy_proxy[step] = base_energy_proxy
            previous_lane_stats = lane_entropy.detach()

        route_distribution = effective_router_weights.mean(dim=0)
        route_entropy = self._distribution_entropy(route_distribution)
        dominant_lane_mass = float(route_distribution.max().item())
        diagnostics = {
            "mean_update_rate": float(hssm_output.update_mask.float().mean().item()),
            "mean_entropy": float(entropy_tensor.mean().item()),
            "embedding_norm": float(embeddings.norm(dim=-1).mean().item()),
            "route_entropy": float(route_entropy),
            "learned_route_entropy": float(self._distribution_entropy(router_weights.mean(dim=0))),
            "dominant_lane_mass": dominant_lane_mass,
            "energy_proxy": float(energy_proxy.mean().item()),
        }
        always_on_energy = float((phase_policy["always_on_pre_mask"].float().to(device) * torch.tensor(
            [
                self.config.model.lane_cost_lm,
                self.config.model.lane_cost_semantic_hot,
                self.config.model.lane_cost_erm,
                self.config.model.lane_cost_semantic_cold,
                self.config.model.lane_cost_eem,
                self.config.model.lane_cost_graph,
            ],
            device=device,
        )).sum().item())
        full_enabled_energy = (
            always_on_energy
            + self.config.model.lane_cost_semantic_cold
            + self.config.model.lane_cost_eem
            + self.config.model.lane_cost_graph
        )
        hot_occupancy = sum(
            len(self.semantic_memory.hot_slots[level]) / max(self.config.semantic_memory.hot_slots, 1)
            for level in range(self.config.hssm.num_levels)
        ) / max(self.config.hssm.num_levels, 1)
        memory_stats = {
            "hot_reads": float(total_hot_reads),
            "cold_reads": float(total_cold_reads),
            "hot_occupancy": float(hot_occupancy),
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
            "graph_fusion_steps": float(graph_fusion_steps),
            "graph_task_eligible_steps": float(graph_task_eligible_steps),
            "cold_semantic_invocations": float(cold_semantic_invocations),
            "eem_invocations": float(eem_invocations),
            "graph_invocations": float(graph_invocations),
            "symbol_link_hits": float(symbol_link_hits),
            "avg_energy_proxy": float(energy_proxy.mean().item()),
            "always_on_energy": float(always_on_energy),
            "full_enabled_energy": float(full_enabled_energy),
            "hard_gated_energy_savings": float(full_enabled_energy - energy_proxy.mean().item()),
            "router_entropy": float(route_entropy),
            "dominant_lane_mass": float(dominant_lane_mass),
            "warmup_steps_remaining": float(warmup_steps_remaining),
            "warmup_active_steps": float(warmup_active_steps),
            "dominant_lane_drop_steps": float(dominant_lane_drop_steps),
            "router_collapse_steps": float(router_collapse_steps),
            "avg_skipped_expensive_reads": float(
                (~router_pre_mask[:, 3:]).float().sum(dim=-1).mean().item()
            ),
        }
        phase_exit_probe_metrics = {
            "recent_copy_hit_rate": float(copy_target_hits / max(seq_len - 1, 1)),
            "episodic_hit_rate": float(episodic_target_hits / max(seq_len - 1, 1)),
            "graph_copy_hit_rate": float(graph_copy_hits / max(seq_len - 1, 1)),
            "symbol_link_hit_rate": float(symbol_link_hits / max(seq_len - 1, 1)),
            "route_entropy": float(route_entropy),
            "energy_proxy": float(energy_proxy.mean().item()),
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
            base_hidden_states=base_hidden_states,
            graph_contexts=graph_contexts,
            router_weights=router_weights,
            effective_router_weights=effective_router_weights,
            oracle_router_weights=oracle_router_weights,
            oracle_availability=oracle_availability,
            router_pre_mask=router_pre_mask,
            router_post_mask=router_post_mask,
            lane_entropies=lane_entropies,
            invoked_lanes=invoked_lanes,
            energy_proxy=energy_proxy,
            warmup_beta=warmup_beta,
            collapse_detected=collapse_detected,
            phase_name=phase_name.value,
            task_label=task_name.value,
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
                "graph_candidate_scores": graph_candidate_scores,
                "graph_target_node_ids": graph_target_node_ids,
                "route_teacher_indices": route_teacher_indices,
                "route_teacher_expensive": route_teacher_expensive,
                "router_pre_logits": router_pre_logits,
                "router_post_logits": router_post_logits,
                "router_post_masks": router_post_masks,
                "oracle_router_weights": oracle_router_weight_list,
                "edit_token_scores": edit_token_scores,
                "task_supervision_mask": None,
                "infill_span": None,
                "maintenance_decision": None,
                "phase_exit_probe_metrics": phase_exit_probe_metrics,
            },
        )

    def _aggregate_semantic_context(
        self,
        master_state: torch.Tensor,
        per_level_outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        gamma = torch.softmax(
            (self.level_gate_vectors @ self.master_norm(master_state))
            / math.sqrt(self.config.model.model_dim),
            dim=0,
        )
        stacked_outputs = torch.stack(per_level_outputs, dim=0)
        projected_outputs = torch.stack(
            [
                self.level_output_projections[level](stacked_outputs[level])
                for level in range(self.config.hssm.num_levels)
            ],
            dim=0,
        )
        return torch.sum(gamma.unsqueeze(-1) * projected_outputs, dim=0)

    def _graph_query_context(self, batch: PhaseABatch, step: int) -> RepoGraphQueryContext:
        structure = batch.document.token_structures[step]
        return RepoGraphQueryContext(
            file_path=self._normalize_graph_path(structure.file_id),
            current_symbol_id=structure.symbol_id,
            current_symbol_name=structure.symbol_name,
            scope_path=structure.scope_path,
            token_value=batch.document.tokens[step].value,
            token_class=batch.document.tokens[step].token_class.value,
        )

    def _router_metadata(
        self,
        batch: PhaseABatch,
        step: int,
        token_counts: Counter[str],
        previous_lane_stats: torch.Tensor,
        availability: tuple[bool, bool, bool],
    ) -> torch.Tensor:
        token = batch.document.tokens[step]
        syntax = batch.document.syntax_features[step]
        rarity = 1.0 / max(token_counts.get(token.value if token.value else token.token_type, 1), 1)
        control_flag = float(token.token_class in {TokenClass.NEWLINE, TokenClass.INDENT, TokenClass.DEDENT})
        features = torch.tensor(
            [
                batch.token_class_ids[step].item() / max(len(TokenClass) - 1, 1),
                rarity,
                float(token.token_class == TokenClass.STRING),
                float(token.token_class == TokenClass.NUMBER),
                float(token.token_class == TokenClass.COMMENT),
                float(token.token_class == TokenClass.KEYWORD),
                control_flag,
                min(float(syntax.depth) / 32.0, 1.0),
                min(float(syntax.block_depth) / 16.0, 1.0),
                float(syntax.inside_call),
                float(syntax.inside_literal),
                float(syntax.inside_comment),
                batch.language_ids[step].item() / 7.0,
                float(availability[0]),
                float(availability[1]),
                float(availability[2]),
                float(previous_lane_stats[0].item()),
                float(previous_lane_stats[1].item()),
                float(previous_lane_stats[2].item()),
                float(previous_lane_stats[3].item()),
                float(previous_lane_stats[4].item()),
            ],
            dtype=torch.float32,
            device=batch.token_ids.device,
        )
        return features

    def _distribution_entropy(self, distribution: torch.Tensor) -> float:
        if distribution.numel() == 0:
            return 0.0
        normalized = distribution / distribution.sum().clamp_min(1e-8)
        return float((-normalized * torch.log(normalized.clamp_min(1e-8))).sum().item())

    def _resolve_phase(self, phase: TrainingPhase | str | None) -> TrainingPhase:
        if phase is None:
            return TrainingPhase(self.config.model.training_phase)
        if isinstance(phase, TrainingPhase):
            return phase
        return TrainingPhase(phase)

    def _resolve_task_label(
        self,
        batch: PhaseABatch,
        task_label: TaskLabel | str | None,
    ) -> TaskLabel:
        if task_label is not None:
            if isinstance(task_label, TaskLabel):
                return task_label
            return TaskLabel(task_label)
        file_name = Path(batch.document.file_path).name.lower()
        if "recent_copy" in file_name:
            return TaskLabel.RECENT_COPY
        if "episodic" in file_name:
            return TaskLabel.EPISODIC_RECALL
        if "edit" in file_name or "patch" in file_name:
            return TaskLabel.EDIT_FIX
        if self.repo_graph_root is not None and self.repo_graph_memory.index is not None:
            normalized = self._normalize_graph_path(batch.document.file_path)
            if normalized in self.repo_graph_memory.index.node_ids_by_file:
                return TaskLabel.REPO_GRAPH
        return TaskLabel.AR

    def _phase_policy(self, phase: TrainingPhase) -> dict[str, torch.Tensor | bool]:
        device = self.level_gate_vectors.device
        if phase == TrainingPhase.PHASE_A:
            return {
                "always_on_pre_mask": torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool, device=device),
                "allowed_post_mask": torch.tensor([0, 1, 0, 0, 0], dtype=torch.bool, device=device),
                "erm_enabled": False,
                "eem_enabled": False,
                "graph_enabled": False,
                "cold_semantic_enabled": False,
                "warmup_enabled": False,
            }
        if phase == TrainingPhase.PHASE_B:
            return {
                "always_on_pre_mask": torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool, device=device),
                "allowed_post_mask": torch.tensor([1, 1, 1, 0, 0], dtype=torch.bool, device=device),
                "erm_enabled": True,
                "eem_enabled": False,
                "graph_enabled": False,
                "cold_semantic_enabled": True,
                "warmup_enabled": True,
            }
        if phase == TrainingPhase.PHASE_C:
            return {
                "always_on_pre_mask": torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool, device=device),
                "allowed_post_mask": torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool, device=device),
                "erm_enabled": True,
                "eem_enabled": True,
                "graph_enabled": False,
                "cold_semantic_enabled": True,
                "warmup_enabled": True,
            }
        if phase == TrainingPhase.PHASE_D:
            return {
                "always_on_pre_mask": torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool, device=device),
                "allowed_post_mask": torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool, device=device),
                "erm_enabled": True,
                "eem_enabled": True,
                "graph_enabled": True,
                "cold_semantic_enabled": True,
                "warmup_enabled": True,
            }
        return {
            "always_on_pre_mask": torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool, device=device),
            "allowed_post_mask": torch.tensor([1, 1, 1, 1, 1], dtype=torch.bool, device=device),
            "erm_enabled": True,
            "eem_enabled": True,
            "graph_enabled": True,
            "cold_semantic_enabled": True,
            "warmup_enabled": False,
        }

    def _oracle_availability(
        self,
        copy_hit: bool,
        episodic_hit: bool,
        graph_hit: bool,
        post_mask: torch.Tensor,
        phase: TrainingPhase,
    ) -> torch.Tensor:
        device = post_mask.device
        availability = torch.tensor(
            [1, 1, int(copy_hit), int(episodic_hit), int(graph_hit)],
            dtype=torch.bool,
            device=device,
        )
        if phase == TrainingPhase.PHASE_A:
            availability = torch.tensor([0, 1, 0, 0, 0], dtype=torch.bool, device=device)
        availability = availability & post_mask
        if not bool(availability.any().item()):
            availability = post_mask.clone()
        return availability

    def _route_teacher(
        self,
        batch: PhaseABatch,
        step: int,
        copy_hit: bool,
        episodic_hit: bool,
        graph_hit: bool,
        phase: TrainingPhase,
    ) -> tuple[int, tuple[int, int, int]]:
        token = batch.document.tokens[step]
        if phase == TrainingPhase.PHASE_A:
            return 1, (0, 0, 0)
        if copy_hit:
            return 2, (0, 0, 0)
        if episodic_hit:
            return 3, (0, 1, 0)
        if graph_hit and phase in {TrainingPhase.PHASE_D, TrainingPhase.PHASE_E}:
            return 4, (0, 0, 1)
        if token.token_class in {
            TokenClass.KEYWORD,
            TokenClass.OPERATOR,
            TokenClass.DELIMITER,
            TokenClass.NEWLINE,
            TokenClass.INDENT,
            TokenClass.DEDENT,
            TokenClass.COMMENT,
        }:
            return 1, (1, 0, 0)
        return 0, (0, 0, 0)

    def _graph_task_is_relevant(self, batch: PhaseABatch, step: int) -> bool:
        if self.repo_graph_memory.index is None:
            return False
        token = batch.document.tokens[step]
        if token.token_class not in {TokenClass.IDENTIFIER, TokenClass.STRING, TokenClass.NUMBER}:
            return False

        context = self._graph_query_context(batch, step)
        if context.current_symbol_id is not None or context.current_symbol_name is not None:
            return True

        import_closure = self.repo_graph_memory.index.import_closure_by_file.get(context.file_path, ())
        test_files = self.repo_graph_memory.index.test_files_by_source.get(context.file_path, ())
        diagnostic_files = self.repo_graph_memory.index.diagnostic_files_by_source.get(context.file_path, ())
        if import_closure or test_files or diagnostic_files:
            return True
        return False

    def _resolve_graph_target_node(
        self,
        target_symbol_id: str | None,
        target_symbol_name: str | None,
        candidate_node_ids: tuple[str, ...],
        candidate_names: tuple[str, ...],
    ) -> str | None:
        if target_symbol_id is not None and target_symbol_id in candidate_node_ids:
            return target_symbol_id
        if target_symbol_name is None:
            return None
        for candidate_id, candidate_name in zip(candidate_node_ids, candidate_names, strict=False):
            if candidate_name == target_symbol_name:
                return candidate_id
        return None

    def _empty_erm_result(self, device: torch.device) -> ExactRecentReadResult:
        return ExactRecentReadResult(
            distribution=torch.zeros(self.config.model.vocabulary_size, device=device),
            log_distribution=torch.full(
                (self.config.model.vocabulary_size,),
                fill_value=math.log(1e-8),
                device=device,
            ),
            attention=torch.zeros(self.config.model.recent_window, device=device),
            slot_token_ids=torch.full(
                (self.config.model.recent_window,),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            ),
            filled_size=0,
            read_count=0,
            write_count=0,
            overwrite_count=0,
        )

    def _empty_eem_result(self, device: torch.device) -> ExactEpisodicReadResult:
        return ExactEpisodicReadResult(
            distribution=torch.zeros(self.config.model.vocabulary_size, device=device),
            log_distribution=torch.full(
                (self.config.model.vocabulary_size,),
                fill_value=math.log(1e-8),
                device=device,
            ),
            chunk_attention=torch.zeros(self.config.model.eem_top_k, device=device),
            pointer_attention=torch.zeros(
                self.config.model.eem_top_k * self.config.model.max_chunk_tokens,
                device=device,
            ),
            retrieved_chunk_ids=torch.full(
                (self.config.model.eem_top_k,),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            ),
            pointer_token_ids=torch.full(
                (self.config.model.eem_top_k * self.config.model.max_chunk_tokens,),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            ),
            retrieved_chunk_count=0,
            read_count=0,
            chunks_finalized=self.exact_episodic_memory.total_chunks_finalized,
            chunk_overhead=0.0,
            stored_chunks=len(self.exact_episodic_memory.chunks),
        )

    def _empty_graph_result(self, device: torch.device) -> RepoGraphReadResult:
        return RepoGraphReadResult(
            graph_context=torch.zeros(self.config.model.graph_value_dim, device=device),
            distribution=torch.zeros(self.config.model.vocabulary_size, device=device),
            log_distribution=torch.full(
                (self.config.model.vocabulary_size,),
                fill_value=math.log(1e-8),
                device=device,
            ),
            attention=torch.zeros(self.config.model.graph_top_k, device=device),
            copy_token_ids=torch.full(
                (self.config.model.graph_top_k,),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            ),
            candidate_scores=torch.zeros(0, device=device),
            candidate_node_ids=(),
            candidate_kinds=(),
            candidate_names=(),
            retrieved_count=0,
            read_count=0,
            candidate_count=0,
            copy_supported_count=0,
            samefile_hits=0,
            import_hits=0,
            symbol_hits=0,
            test_hits=0,
            diagnostic_hits=0,
            target_node_id=None,
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
