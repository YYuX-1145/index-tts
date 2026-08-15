
import os
import torch
import time
from typing import Any, List, Optional, Tuple, Union

from packaging import version
import importlib

# FlashInfer's sampler JIT invokes ``link.exe`` on Windows.  Some packaged
# environments put GNU coreutils' unrelated link.exe before MSVC's linker,
# making engine profiling fail.  vLLM's native sampler avoids that JIT and is
# the supported fallback selected by this environment variable.
if os.name == "nt":
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

vllm_version = version.parse(importlib.import_module("vllm").__version__)

# 在 vllm 中注册自定义的 GPT2TTSModel
from vllm import ModelRegistry

ModelRegistry.register_model(
    "GPT2InferenceModel",
    "indextts.gpt.index_tts_gpt2_vllm_v1:GPT2TTSModel",
)
print("✅ Registered GPT2TTSModel with vLLM")


# 将 position_ids 减去 prefill 的长度再加 1，以便正确计算每一步 decode 的 position embedding
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
import numpy as np
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder

def _prepare_inputs(
    self,
    scheduler_output: "SchedulerOutput",
) -> tuple[dict[str, Any], torch.Tensor, Optional[SpecDecodeMetadata],
            np.ndarray, Optional[CommonAttentionMetadata], int]:
    """
    :return: tuple[
        attn_metadata: layer-to-attention_metadata mapping,
        logits_indices, spec_decode_metadata
    ]
    """
    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    assert total_num_scheduled_tokens > 0
    num_reqs = self.input_batch.num_reqs
    assert num_reqs > 0

    # OPTIMIZATION: Start copying the block table first.
    # This way, we can overlap the copy with the following CPU operations.
    self.input_batch.block_table.commit_block_table(num_reqs)

    # Get the number of scheduled tokens for each request.
    req_ids = self.input_batch.req_ids
    tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
    num_scheduled_tokens = np.array(tokens, dtype=np.int32)
    max_num_scheduled_tokens = max(tokens)

    # Get request indices.
    # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
    req_indices = np.repeat(self.arange_np[:num_reqs],
                            num_scheduled_tokens)

    # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
    # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    cu_num_tokens, arange = self._get_cumsum_and_arange(
        num_scheduled_tokens)

    # Get positions.
    positions_np = self.positions.np[:total_num_scheduled_tokens]
    np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
            arange,
            out=positions_np)

    # Calculate M-RoPE positions.
    # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
    if self.uses_mrope:
        self._calc_mrope_positions(scheduler_output)

    # Get token indices.
    # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
    # where M is the max_model_len.
    token_indices = (positions_np +
                        req_indices * self.input_batch.token_ids_cpu.shape[1])

    # NOTE(woosuk): We use torch.index_select instead of np.take here
    # because torch.index_select is much faster than np.take for large
    # tensors.
    torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                        0,
                        torch.from_numpy(token_indices),
                        out=self.input_ids.cpu[:total_num_scheduled_tokens])

    self.input_batch.block_table.compute_slot_mapping(
        req_indices, positions_np)
    self.input_batch.block_table.commit_slot_mapping(
        total_num_scheduled_tokens)

    # GPT2TTSModel position ids support
    model = self.get_model()
    if isinstance(model, GPT2TTSModel):
        # req_ids_in_batch = self.input_batch.req_ids[:num_reqs]
        prompt_tokens_offset = []
        for req_id in self.input_batch.req_ids:
            prompt_tokens_offset.append(-(len(self.requests[req_id].prompt_token_ids) - 1))
            # print(f"[{idx}] self.requests[req_id].prompt_token_ids:", len(self.requests[req_id].prompt_token_ids), positions_np)
        np.add(np.array(prompt_tokens_offset)[req_indices],
                positions_np,
                out=positions_np)

    # Prepare the attention metadata.
    self.query_start_loc.np[0] = 0
    self.query_start_loc.np[1:num_reqs + 1] = cu_num_tokens
    # Note: pad query_start_loc to be non-decreasing, as kernels
    # like FlashAttention requires that
    self.query_start_loc.np[num_reqs + 1:].fill(cu_num_tokens[-1])
    self.query_start_loc.copy_to_gpu()
    query_start_loc = self.query_start_loc.gpu[:num_reqs + 1]

    self.seq_lens.np[:num_reqs] = (
        self.input_batch.num_computed_tokens_cpu[:num_reqs] +
        num_scheduled_tokens)
    # Fill unused with 0 for full cuda graph mode.
    self.seq_lens.np[num_reqs:].fill(0)
    self.seq_lens.copy_to_gpu()
    seq_lens = self.seq_lens.gpu[:num_reqs]
    max_seq_len = self.seq_lens.np[:num_reqs].max().item()

    # Copy the tensors to the GPU.
    self._prepare_input_ids(total_num_scheduled_tokens, cu_num_tokens)

    if self.uses_mrope:
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
            self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
            non_blocking=True)
    else:
        # Common case (1D positions)
        self.positions.copy_to_gpu(total_num_scheduled_tokens)

    use_spec_decode = len(
        scheduler_output.scheduled_spec_decode_tokens) > 0
    if not use_spec_decode:
        # NOTE(woosuk): Due to chunked prefills, the batch may contain
        # partial requests. While we should not sample any token
        # from these partial requests, we do so for simplicity.
        # We will ignore the sampled tokens from the partial requests.
        # TODO: Support prompt logprobs.
        logits_indices = query_start_loc[1:] - 1
        num_draft_tokens = None
        spec_decode_metadata = None
    else:
        # Get the number of draft tokens for each request.
        # Iterate over the dictionary rather than all requests since not all
        # requests have draft tokens.
        num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
        for req_id, draft_token_ids in (
                scheduler_output.scheduled_spec_decode_tokens.items()):
            req_idx = self.input_batch.req_id_to_index[req_id]
            num_draft_tokens[req_idx] = len(draft_token_ids)

        spec_decode_metadata = self._calc_spec_decode_metadata(
            num_draft_tokens, cu_num_tokens)
        logits_indices = spec_decode_metadata.logits_indices
        self.num_draft_tokens.np[:num_reqs] = num_draft_tokens
        self.num_draft_tokens.np[num_reqs:].fill(0)
        self.num_draft_tokens.copy_to_gpu()

    logits_indices_padded = None
    if self.cache_config.kv_sharing_fast_prefill:
        logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
            logits_indices)

    attn_metadata: dict[str, Any] = {}

    # Used in the below loop.
    query_start_loc_cpu = self.query_start_loc.cpu[:num_reqs + 1]
    seq_lens_cpu = self.seq_lens.cpu[:num_reqs]
    num_computed_tokens_cpu = (
        self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs])
    spec_decode_common_attn_metadata = None
    if use_spec_decode:
        self.num_accepted_tokens.np[:num_reqs] = (
            self.input_batch.num_accepted_tokens_cpu[:num_reqs])
        self.num_accepted_tokens.np[num_reqs:].fill(1)
        self.num_accepted_tokens.copy_to_gpu()

    # Prepare the attention metadata for each KV cache group and make layers
    # in the same group share the same metadata.
    for kv_cache_group_id, kv_cache_group_spec in enumerate(
            self.kv_cache_config.kv_cache_groups):
        encoder_seq_lens = self._get_encoder_seq_lens(
            scheduler_output, kv_cache_group_spec.kv_cache_spec, num_reqs)

        if isinstance(kv_cache_group_spec.kv_cache_spec,
                        EncoderOnlyAttentionSpec):
            # Encoder-only layers do not have KV cache, so we need to
            # create a dummy block table and slot mapping for them.
            blk_table_tensor = torch.zeros(
                (num_reqs, 1),
                dtype=torch.int32,
                device=self.device,
            )
            slot_mapping = torch.zeros(
                (total_num_scheduled_tokens, ),
                dtype=torch.int64,
                device=self.device,
            )
            num_common_prefix_blocks = 0
        else:
            blk_table = self.input_batch.block_table[kv_cache_group_id]
            blk_table_tensor = blk_table.get_device_tensor()[:num_reqs]
            slot_mapping = blk_table.slot_mapping[:
                                                    total_num_scheduled_tokens]

            # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # graph mode.
            blk_table.slot_mapping[total_num_scheduled_tokens:].fill_(-1)
            num_common_prefix_blocks = (
                scheduler_output.
                num_common_prefix_blocks[kv_cache_group_id])

        common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            num_reqs=num_reqs,
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
            max_seq_len=max_seq_len,
            block_table_tensor=blk_table_tensor,
            slot_mapping=slot_mapping,
            logits_indices_padded=logits_indices_padded,
            num_logits_indices=logits_indices.size(0),
            causal=True,
            encoder_seq_lens=encoder_seq_lens,
        )

        if self.speculative_config and \
            spec_decode_common_attn_metadata is None:
            spec_decode_common_attn_metadata = common_attn_metadata

        for attn_group in self.attn_groups[kv_cache_group_id]:
            # Prepare for cascade attention if enabled & beneficial.
            common_prefix_len = 0
            builder = attn_group.metadata_builder
            if self.cascade_attn_enabled:
                common_prefix_len = self._compute_cascade_attn_prefix_len(
                    num_scheduled_tokens,
                    num_common_prefix_blocks,
                    kv_cache_group_spec.kv_cache_spec,
                    builder,
                )

            extra_attn_metadata_args = {}
            if use_spec_decode and isinstance(builder,
                                                GDNAttentionMetadataBuilder):
                extra_attn_metadata_args = dict(
                    num_accepted_tokens=self.num_accepted_tokens.
                    gpu[:num_reqs],
                    num_draft_tokens=self.num_draft_tokens.gpu[:num_reqs],
                )

            attn_metadata_i = builder.build(
                common_prefix_len=common_prefix_len,
                common_attn_metadata=common_attn_metadata,
                **extra_attn_metadata_args)

            for layer_name in attn_group.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

    # Hot-Swap lora model
    if self.lora_config:
        self.set_active_loras(self.input_batch, num_scheduled_tokens)

    return (attn_metadata, logits_indices, spec_decode_metadata,
            num_scheduled_tokens, spec_decode_common_attn_metadata,
            max_num_scheduled_tokens)

# Keep vLLM's current implementation intact and only apply IndexTTS' custom
# position convention afterwards.  The full method is internal API and its
# signature/return value changed substantially in vLLM 0.24.
_vllm_prepare_inputs = GPUModelRunner._prepare_inputs


def _prepare_inputs_vllm_024(
    self,
    scheduler_output: "SchedulerOutput",
    num_scheduled_tokens: np.ndarray,
):
    result = _vllm_prepare_inputs(
        self, scheduler_output, num_scheduled_tokens
    )

    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    num_reqs = self.input_batch.num_reqs
    req_indices = np.repeat(
        self.arange_np[:num_reqs], num_scheduled_tokens
    )
    prompt_offsets = torch.tensor(
        [
            -(len(self.requests[req_id].prompt_token_ids) - 1)
            for req_id in self.input_batch.req_ids
        ],
        dtype=self.positions.dtype,
        device=self.positions.device,
    )
    req_indices_gpu = torch.as_tensor(
        req_indices, dtype=torch.long, device=self.positions.device
    )
    self.positions[:total_num_scheduled_tokens].add_(
        prompt_offsets[req_indices_gpu]
    )
    return result


GPUModelRunner._prepare_inputs = _prepare_inputs_vllm_024
print("✅ Patched GPUModelRunner._prepare_inputs")


