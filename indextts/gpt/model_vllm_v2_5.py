"""IndexTTS-2.5 GPT prompt/conditioning adapter for vLLM.

As in the V2 vLLM path, the autoregressive model is loaded by vLLM from the
separately converted ``model_dir/gpt`` directory.  This module restores the
non-backbone weights from ``gpt.pth`` and builds the official V2.5 prompt.
"""

import uuid
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import patch_vllm  # noqa: F401 - registers GPT2InferenceModel with vLLM
from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler
from indextts.utils.tokenizer import LANGUAGE_DICT
from vllm import SamplingParams, TokensPrompt
from vllm.v1.engine.async_llm import AsyncLLM

from indextts.gpt.index_tts_gpt2_vllm_v1 import PLACEHOLDER_TOKEN_ID


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len: int, model_dim: int, init: float = 0.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(torch.arange(0, x.shape[1], device=x.device))


class UnifiedVoice(nn.Module):
    """Official IndexTTS-2.5 conditioning stack with a vLLM GPT backend."""

    def __init__(
        self,
        vllm_model: AsyncLLM,
        layers: int = 8,
        model_dim: int = 512,
        heads: int = 8,
        max_text_tokens: int = 120,
        max_mel_tokens: int = 250,
        max_conditioning_inputs: int = 1,
        mel_length_compression: int = 1024,
        number_text_tokens: int = 256,
        start_text_token: int = 0,
        stop_text_token: int = 1,
        number_mel_codes: int = 8194,
        start_mel_token: int = 8192,
        stop_mel_token: int = 8193,
        train_solo_embeddings: bool = False,
        use_mel_codes_as_input: bool = True,
        checkpointing: bool = True,
        types: int = 1,
        condition_num_latent: int = 32,
        condition_type: str = "conformer_perceiver",
        condition_module: dict[str, Any] | None = None,
        emo_condition_module: dict[str, Any] | None = None,
        **_kwargs: Any,
    ):
        super().__init__()
        if not use_mel_codes_as_input:
            raise ValueError("IndexTTS-2.5 vLLM requires discrete mel codes")
        if emo_condition_module is None:
            raise ValueError("emo_condition_module is required")

        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.condition_type = condition_type
        self.cond_num = condition_num_latent
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)

        # IndexTTS-2.5 always uses CAMPPlus speaker conditioning.
        self.spk_emb_proj = nn.Linear(192, model_dim)
        self.emo_conditioning_encoder = ConformerEncoder(
            input_size=1024,
            output_size=emo_condition_module["output_size"],
            linear_units=emo_condition_module["linear_units"],
            attention_heads=emo_condition_module["attention_heads"],
            num_blocks=emo_condition_module["num_blocks"],
            input_layer=emo_condition_module["input_layer"],
        )
        self.emo_perceiver_encoder = PerceiverResampler(
            1024,
            dim_context=emo_condition_module["output_size"],
            ff_mult=emo_condition_module["perceiver_mult"],
            heads=emo_condition_module["attention_heads"],
            num_latents=1,
        )

        self.text_embedding = nn.Embedding(
            number_text_tokens * types + 1, model_dim
        )
        self.lang_embedding = nn.Embedding(len(LANGUAGE_DICT) + 1, model_dim)
        self.emo_layer = nn.Linear(model_dim, model_dim)
        self.emovec_layer = nn.Linear(1024, model_dim)
        self.mel_embedding = nn.Embedding(number_mel_codes, model_dim)

        max_mel_seq_len = max_mel_tokens + 2 + max_conditioning_inputs
        max_text_seq_len = max_text_tokens + 2
        self.mel_pos_embedding = LearnedPositionEmbeddings(
            max_mel_seq_len, model_dim
        )
        self.text_pos_embedding = LearnedPositionEmbeddings(
            max_text_seq_len, model_dim
        )
        self.mel_solo_embedding = 0
        self.text_solo_embedding = 0

        # Deliberately no transformers.GPT2Model here. The backbone is owned
        # exclusively by vLLM. Its final norm and LM head live there as well;
        # this adapter only keeps modules used to construct the input prompt.
        self.gpt = None
        self.llm = vllm_model

    def get_emo_conditioning(
        self,
        speech_conditioning_input: torch.Tensor,
        cond_mel_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        speech_conditioning_input, mask = self.emo_conditioning_encoder(
            speech_conditioning_input.transpose(1, 2), cond_mel_lengths
        )
        conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
        conds = self.emo_perceiver_encoder(
            speech_conditioning_input, conds_mask
        )
        return conds.squeeze(1)

    def get_emovec(
        self,
        emo_speech_conditioning_latent: torch.Tensor,
        emo_cond_lengths: torch.Tensor,
    ) -> torch.Tensor:
        emo_vec = self.get_emo_conditioning(
            emo_speech_conditioning_latent.transpose(1, 2),
            emo_cond_lengths,
        )
        return self.emo_layer(self.emovec_layer(emo_vec))

    def merge_emovec(
        self,
        speech_conditioning_latent: torch.Tensor,
        emo_speech_conditioning_latent: torch.Tensor,
        cond_lengths: torch.Tensor,
        emo_cond_lengths: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        emo_vec = self.get_emovec(
            emo_speech_conditioning_latent, emo_cond_lengths
        )
        base_vec = self.get_emovec(speech_conditioning_latent, cond_lengths)
        return base_vec + alpha * (emo_vec - base_vec)

    def prepare_gpt_inputs(
        self,
        conditional_latents: torch.Tensor,
        text_inputs: torch.Tensor,
        langs: torch.Tensor,
    ) -> torch.Tensor:
        """Build the exact official V2.5 [condition][text][MEL_START] prompt."""
        if conditional_latents.shape[0] != text_inputs.shape[0]:
            if conditional_latents.shape[0] != 1:
                raise ValueError("Conditioning batch is incompatible with text batch")
            conditional_latents = conditional_latents.expand(
                text_inputs.shape[0], -1, -1
            )

        prompts = []
        for index, text_input in enumerate(text_inputs):
            # The official tokenizer already appends stop=1. Remove wrapper
            # tokens and add exactly one start/stop pair around the content.
            text_input = text_input[
                (text_input != self.start_text_token)
                & (text_input != self.stop_text_token)
            ]
            text_input = F.pad(
                text_input, (1, 0), value=self.start_text_token
            )
            text_input = F.pad(
                text_input, (0, 1), value=self.stop_text_token
            )
            text_positions = torch.arange(
                text_input.numel(), device=text_input.device
            )
            text_emb = self.text_embedding(text_input)
            text_emb = text_emb + self.text_pos_embedding.emb(text_positions)
            text_emb = text_emb + self.lang_embedding(langs[index])

            prompt = torch.cat(
                [conditional_latents[index], text_emb], dim=0
            )
            mel_start = torch.tensor(
                [self.start_mel_token],
                dtype=torch.long,
                device=text_input.device,
            )
            mel_start_emb = self.mel_embedding(mel_start)
            mel_start_emb = mel_start_emb + self.mel_pos_embedding.emb(
                torch.zeros(1, dtype=torch.long, device=text_input.device)
            )
            prompts.append(torch.cat([prompt, mel_start_emb], dim=0))

        if len(prompts) != 1:
            raise ValueError(
                "IndexTTS-2.5 vLLM currently accepts one text segment per request"
            )
        return prompts[0].unsqueeze(0)

    async def inference_speech(
        self,
        speech_condition: torch.Tensor,
        text_inputs: torch.Tensor,
        langs: torch.Tensor,
        emo_speech_condition: torch.Tensor | None = None,
        cond_lengths: torch.Tensor | None = None,
        emo_cond_lengths: torch.Tensor | None = None,
        emo_vec: torch.Tensor | None = None,
        use_speed: bool = False,
        use_campplus: bool = True,
        wav: str | None = None,
        campplus_embedding: torch.Tensor | None = None,
        input_tokens: torch.Tensor | None = None,
        num_return_sequences: int = 1,
        max_generate_length: int | None = None,
        typical_sampling: bool = False,
        typical_mass: float = 0.9,
        autocast_dtype: torch.dtype | None = None,
        **generation_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del use_speed, use_campplus, wav, typical_sampling, typical_mass
        if input_tokens is not None:
            raise NotImplementedError("V2.5 vLLM does not support input_tokens")
        if num_return_sequences != 1:
            raise NotImplementedError(
                "V2.5 vLLM currently supports num_return_sequences=1"
            )
        if campplus_embedding is None:
            raise ValueError("IndexTTS-2.5 requires a CAMPPlus embedding")

        # Torch grad/inference/autocast modes are thread-local rather than
        # asyncio-task-local. Keep this context entirely before vLLM's async
        # generation so concurrent requests cannot restore each other's mode.
        with torch.no_grad():
            with torch.amp.autocast(
                text_inputs.device.type,
                enabled=autocast_dtype is not None,
                dtype=autocast_dtype,
            ):
                if speech_condition.ndim == 2:
                    speech_condition = speech_condition.unsqueeze(0)
                if emo_speech_condition is None:
                    emo_speech_condition = speech_condition
                if cond_lengths is None:
                    cond_lengths = torch.tensor(
                        [speech_condition.shape[-1]],
                        device=speech_condition.device,
                    )
                if emo_cond_lengths is None:
                    emo_cond_lengths = torch.tensor(
                        [emo_speech_condition.shape[-1]],
                        device=emo_speech_condition.device,
                    )

                speech_conditioning_latent = self.spk_emb_proj(
                    campplus_embedding
                )
                if speech_conditioning_latent.ndim != 3:
                    speech_conditioning_latent = (
                        speech_conditioning_latent.unsqueeze(1)
                    )

                if emo_vec is None:
                    emo_vec = self.get_emovec(
                        emo_speech_condition, emo_cond_lengths
                    )
                zero_controls = torch.zeros(
                    speech_conditioning_latent.shape[0],
                    2,
                    speech_conditioning_latent.shape[2],
                    dtype=speech_conditioning_latent.dtype,
                    device=speech_conditioning_latent.device,
                )
                conditional_latents = torch.cat(
                    [
                        speech_conditioning_latent + emo_vec.unsqueeze(1),
                        zero_controls,
                    ],
                    dim=1,
                )
                inputs_embeds = self.prepare_gpt_inputs(
                    conditional_latents, text_inputs, langs
                )

        do_sample = bool(generation_kwargs.pop("do_sample", True))
        temperature = float(generation_kwargs.pop("temperature", 0.8))
        if not do_sample:
            temperature = 0.0
        top_p = float(generation_kwargs.pop("top_p", 0.8))
        top_k_value = generation_kwargs.pop("top_k", 30)
        top_k = int(top_k_value) if top_k_value is not None else 0
        repetition_penalty = float(
            generation_kwargs.pop("repetition_penalty", 10.0)
        )
        # vLLM uses sampling rather than the official Transformers beam search.
        generation_kwargs.pop("num_beams", None)
        generation_kwargs.pop("length_penalty", None)
        max_tokens = min(
            int(max_generate_length or self.max_mel_tokens),
            self.max_mel_tokens,
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            stop_token_ids=[self.stop_mel_token],
            include_stop_str_in_output=True,
            detokenize=False,
        )

        prompt = TokensPrompt(
            prompt_token_ids=[PLACEHOLDER_TOKEN_ID],
            multi_modal_data={
                "audio": {
                    "audio_embeds": [
                        inputs_embeds.detach().squeeze(0).cpu()
                    ]
                }
            },
        )
        output_generator = self.llm.generate(
            prompt,
            sampling_params=sampling_params,
            request_id=uuid.uuid4().hex,
        )
        output = None
        async for output in output_generator:
            pass
        if output is None or not output.outputs:
            raise RuntimeError("vLLM returned no IndexTTS-2.5 output")

        codes = torch.tensor(
            output.outputs[0].token_ids,
            dtype=torch.long,
            device=text_inputs.device,
        ).unsqueeze(0)
        return codes, speech_conditioning_latent
