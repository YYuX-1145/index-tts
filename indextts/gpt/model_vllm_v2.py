import uuid
import os
import functools
import patch_vllm  # ⚠️ Monkey Patch, do not delete this line

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessorList

from transformers import GPT2Config, GPT2Model

from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler

from indextts.gpt.index_tts_gpt2_vllm_v1 import PLACEHOLDER_TOKEN, PLACEHOLDER_TOKEN_ID

from vllm import AsyncLLMEngine, SamplingParams, TokensPrompt
from vllm.v1.engine.async_llm import AsyncLLM


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)
    

class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


class UnifiedVoice(nn.Module):
    def __init__(self, vllm_model,
                 layers=8, model_dim=512, heads=8, max_text_tokens=120, max_mel_tokens=250, max_conditioning_inputs=1,
                 mel_length_compression=1024, number_text_tokens=256,
                 start_text_token=0, stop_text_token=1, number_mel_codes=8194, start_mel_token=8192, stop_mel_token=8193,
                 train_solo_embeddings=False, use_mel_codes_as_input=True,
                 checkpointing=True, types=1,
                 condition_num_latent=32, condition_type="perceiver", condition_module=None, emo_condition_module=None):
        """
        Args:
            layers: Number of layers in transformer stack.
            model_dim: Operating dimensions of the transformer
            heads: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64
            max_text_tokens: Maximum number of text tokens that will be encountered by model.
            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
            max_conditioning_inputs: Maximum number of conditioning inputs provided to the model. If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).
            mel_length_compression: The factor between <number_input_samples> and <mel_tokens>. Used to compute MEL code padding given wav input length.
            number_text_tokens:
            start_text_token:
            stop_text_token:
            number_mel_codes:
            start_mel_token:
            stop_mel_token:
            checkpointing:
        """
        super().__init__()
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
        self.cond_num = condition_num_latent
        self.cond_mask_pad = nn.ConstantPad1d((self.cond_num, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)

        self.conditioning_encoder = ConformerEncoder(input_size=1024,
                                                        output_size=condition_module['output_size'],
                                                        linear_units=condition_module['linear_units'],
                                                        attention_heads=condition_module['attention_heads'],
                                                        num_blocks=condition_module['num_blocks'],
                                                        input_layer=condition_module['input_layer'])
        self.perceiver_encoder = PerceiverResampler(model_dim, dim_context=condition_module['output_size'],
                                                    ff_mult=condition_module['perceiver_mult'],
                                                    heads=condition_module['attention_heads'],
                                                    num_latents=self.cond_num)

        self.emo_conditioning_encoder = ConformerEncoder(input_size=1024,
                                                         output_size=emo_condition_module['output_size'],
                                                         linear_units=emo_condition_module['linear_units'],
                                                         attention_heads=emo_condition_module['attention_heads'],
                                                         num_blocks=emo_condition_module['num_blocks'],
                                                         input_layer=emo_condition_module['input_layer'])
        self.emo_perceiver_encoder = PerceiverResampler(1024, dim_context=emo_condition_module['output_size'],
                                                            ff_mult=emo_condition_module['perceiver_mult'],
                                                            heads=emo_condition_module['attention_heads'],
                                                            num_latents=1)

        self.text_embedding = nn.Embedding(self.number_text_tokens * types + 1, model_dim)
        self.emo_layer = nn.Linear(model_dim, model_dim)
        self.emovec_layer = nn.Linear(1024, model_dim)
        self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)

        max_mel_seq_len = self.max_mel_tokens + 2 + self.max_conditioning_inputs
        max_text_seq_len = self.max_text_tokens + 2
        gpt_config = GPT2Config(vocab_size=256,  # Unused.
                                n_positions=max_mel_seq_len + max_text_seq_len,
                                n_ctx=max_mel_seq_len + max_text_seq_len,
                                n_embd=model_dim,
                                n_layer=layers,
                                n_head=heads,
                                gradient_checkpointing=False,
                                use_cache=True)
        self.gpt = GPT2Model(gpt_config)
        # Override the built in positional embeddings
        del self.gpt.wpe
        self.gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
        # Built-in token embeddings are unused.
        del self.gpt.wte
        self.mel_pos_embedding, self.text_pos_embedding  = LearnedPositionEmbeddings(max_mel_seq_len, model_dim), LearnedPositionEmbeddings(max_text_seq_len, model_dim)

        self.mel_solo_embedding = 0
        self.text_solo_embedding = 0

        self.final_norm = nn.LayerNorm(model_dim)  # , dtype=torch.float16
        self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

        self.speed_emb = nn.Embedding(2, model_dim)
        self.speed_emb.weight.data.normal_(mean=0.0, std=0.0)

        # Initialize the embeddings per the GPT-2 scheme
        embeddings = [self.text_embedding, self.mel_embedding]
        for module in embeddings:
            module.weight.data.normal_(mean=0.0, std=.02)

        self.llm: AsyncLLM = vllm_model
        self.sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.8,
            top_k=30,  # 5, 30
            repetition_penalty=10.0,  # 8.0
            max_tokens=2048,  # 605
            stop_token_ids=[self.stop_mel_token],
            include_stop_str_in_output=True,
        )

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, mel_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(mel_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = mel_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def set_text_padding(self, text_input_tokens, text_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(text_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = text_lengths[b]
            if actual_end < text_input_tokens.shape[-1]:
                text_input_tokens[b, actual_end:] = self.stop_text_token
        return text_input_tokens

    def get_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        speech_conditioning_input, mask = self.conditioning_encoder(speech_conditioning_input.transpose(1, 2),
                                                                    cond_mel_lengths)  # (b, s, d), (b, 1, s)
        conds_mask = self.cond_mask_pad(mask.squeeze(1))
        conds = self.perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 32, d)
        return conds

    def get_emo_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        speech_conditioning_input, mask = self.emo_conditioning_encoder(speech_conditioning_input.transpose(1, 2),
                                                                        cond_mel_lengths)  # (b, s, d), (b, 1, s)
        conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
        conds = self.emo_perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 1, d)
        return conds.squeeze(1)

    async def inference_speech(self, speech_condition, text_inputs, emo_speech_condition=None, cond_lengths=None, emo_cond_lengths=None, emo_vec=None, use_speed=False):
        if speech_condition.ndim == 2:
            speech_condition = speech_condition.unsqueeze(0)
        if emo_speech_condition is None:
            emo_speech_condition = speech_condition
        if cond_lengths is None:
            cond_lengths = torch.tensor([speech_condition.shape[-1]], device=speech_condition.device)
        if emo_cond_lengths is None:
            emo_cond_lengths = torch.tensor([emo_speech_condition.shape[-1]], device=speech_condition.device) 

        speech_conditioning_latent = self.get_conditioning(speech_condition.transpose(1,2), cond_lengths)
        if emo_vec is None:
            print('compute emo vec')
            emo_vec = self.get_emo_conditioning(emo_speech_condition.transpose(1,2), emo_cond_lengths)
            emo_vec = self.emovec_layer(emo_vec)
            emo_vec = self.emo_layer(emo_vec)
        else:
            print('Use the specified emotion vector')

        tmp = torch.zeros(text_inputs.size(0)).to(text_inputs.device)
        duration_emb =  self.speed_emb(torch.zeros_like(tmp).long())
        duration_emb_half = self.speed_emb(torch.ones_like(tmp).long())
        conds_latent = torch.cat((speech_conditioning_latent + emo_vec.unsqueeze(1), duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)), 1)
        
        text_inputs, _ = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        emb = torch.cat([conds_latent, text_emb], dim=1)

        mel_start_emb = self.mel_embedding(torch.full((emb.shape[0], 1,), fill_value=self.start_mel_token, dtype=torch.long, device=text_inputs.device))
        mel_start_emb = mel_start_emb + self.mel_pos_embedding(mel_start_emb)
        inputs_embeds = torch.cat([emb, mel_start_emb], dim=1)

        fake_inputs = PLACEHOLDER_TOKEN * 1  # [PLACEHOLDER_TOKEN_ID]
        multi_modal_data = {"audio": {"audio_embeds": [inputs_embeds.squeeze(0).cpu()]}}
        tokens_prompt = TokensPrompt(prompt_token_ids=fake_inputs, multi_modal_data=multi_modal_data)
        # tokens_prompt = TokensPrompt(prompt_token_ids=fake_inputs, multi_modal_data=multi_modal_data)
        output_generator = self.llm.generate(tokens_prompt, sampling_params=self.sampling_params, request_id=uuid.uuid4().hex)
        # latent = []
        async for output in output_generator:
            # latent.append(output.hidden_states.clone())
            pass
        codes = output.outputs[0].token_ids[:-2]
        codes = torch.tensor(codes, device=text_inputs.device, dtype=torch.long).unsqueeze(0)

        return codes, speech_conditioning_latent

    def forward(self, speech_conditioning_latent, text_inputs, text_lengths, mel_codes, mel_codes_lengths, emo_speech_conditioning_latent,
                cond_mel_lengths=None, emo_cond_mel_lengths=None, emo_vec=None, use_speed=None, do_spk_cond=False):
        # TODO: 注意这里的speech_conditioning_latent.transpose(1,2)，与v1不同，先支持一个参考音频，run起来先
        if do_spk_cond:
            speech_conditioning_latent = self.get_conditioning(speech_conditioning_latent.transpose(1,2), cond_mel_lengths)
        else:
            speech_conditioning_latent = speech_conditioning_latent

        if emo_vec is None:
            emo_vec_syn_ori = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1,2), emo_cond_mel_lengths)
            emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
            emo_vec = self.emo_layer(emo_vec_syn)

        text_inputs = self.set_text_padding(text_inputs, text_lengths)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)

        mel_codes = self.set_mel_padding(mel_codes, mel_codes_lengths)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)

        duration_emb = self.speed_emb(torch.zeros_like(use_speed))
        duration_emb_half = self.speed_emb(torch.ones_like(use_speed))
        conds = torch.cat((speech_conditioning_latent + emo_vec.unsqueeze(1), duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)), 1)
        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(mel_codes, self.start_mel_token, self.stop_mel_token)
        
        mel_emb = self.mel_embedding(mel_codes)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)

        emb = torch.cat([conds, text_emb, mel_emb], dim=1)
        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True)
        offset = conds.shape[1]
        enc = gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm(enc)
        
        return enc[:, -mel_emb.shape[1]:][:, :-2]

    def get_emovec(self, emo_speech_conditioning_latent, emo_cond_lengths):
        emo_vec_syn_ori = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1,2), emo_cond_lengths)
        emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
        emo_vec = self.emo_layer(emo_vec_syn)
        return emo_vec

    def merge_emovec(self, speech_conditioning_latent, emo_speech_conditioning_latent, cond_lengths, emo_cond_lengths, alpha = 1.0):
        emo_vec = self.get_emovec(emo_speech_conditioning_latent, emo_cond_lengths)
        base_vec = self.get_emovec(speech_conditioning_latent, cond_lengths)

        out = base_vec + alpha * (emo_vec - base_vec)
        return out
