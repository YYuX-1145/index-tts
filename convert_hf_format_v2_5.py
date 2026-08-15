"""Convert an official IndexTTS-2.5 GPT checkpoint for the vLLM backend."""

import argparse
import json
import os

import torch
from torch import nn
from omegaconf import OmegaConf
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Model,
    GPT2PreTrainedModel,
)

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)


class GPT2InferenceModel(GPT2PreTrainedModel):
    """Inference-only layout expected by GPT2TTSModel in patch_vllm.py."""

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        del self.transformer.wte
        del self.transformer.wpe
        self.text_pos_embedding = LearnedPositionEmbeddings(
            config.n_positions,
            config.n_embd,
        )
        self.audio_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.final_norm = nn.LayerNorm(
            config.n_embd,
            eps=config.layer_norm_epsilon,
        )
        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
        )


def get_source_key(output_key):
    prefixes = {
        "transformer.": "gpt.",
        "text_pos_embedding.": "mel_pos_embedding.",
        "audio_emb.": "mel_embedding.",
        "final_norm.": "final_norm.",
        "lm_head.": "mel_head.",
    }
    for output_prefix, source_prefix in prefixes.items():
        if output_key.startswith(output_prefix):
            return source_prefix + output_key.removeprefix(output_prefix)
    raise KeyError(f"No checkpoint mapping for output parameter: {output_key}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert the official IndexTTS-2.5 gpt.pth checkpoint into the "
            "Hugging Face layout loaded by the vLLM backend."
        )
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the official config.yaml and gpt.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <model_dir>/gpt)",
    )
    parser.add_argument(
        "--dtype",
        choices=DTYPES,
        default="bfloat16",
        help="Export precision (default: bfloat16, matching IndexTTS-2.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used during conversion (default: cpu)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = os.path.abspath(args.model_dir)
    output_dir = os.path.abspath(
        args.output_dir or os.path.join(model_dir, "gpt")
    )
    cfg_path = os.path.join(model_dir, "config.yaml")

    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"IndexTTS-2.5 config not found: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    gpt_path = os.path.join(model_dir, cfg.gpt_checkpoint)
    if not os.path.isfile(gpt_path):
        raise FileNotFoundError(f"IndexTTS-2.5 GPT checkpoint not found: {gpt_path}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    max_mel_positions = (
        cfg.gpt.max_mel_tokens + 2 + cfg.gpt.get("max_conditioning_inputs", 1)
    )
    total_context = cfg.gpt.max_mel_tokens + cfg.gpt.max_text_tokens + 2
    inference_config = GPT2Config(
        vocab_size=cfg.gpt.number_mel_codes,
        n_positions=max_mel_positions,
        n_ctx=total_context,
        n_embd=cfg.gpt.model_dim,
        n_layer=cfg.gpt.layers,
        n_head=cfg.gpt.heads,
        gradient_checkpointing=False,
        use_cache=True,
        bos_token_id=cfg.gpt.start_mel_token,
        eos_token_id=cfg.gpt.stop_mel_token,
        pad_token_id=cfg.gpt.stop_mel_token,
    )
    gpt = GPT2InferenceModel(inference_config)

    checkpoint = torch.load(gpt_path, map_location="cpu", weights_only=True)
    checkpoint = checkpoint.get("model", checkpoint)
    converted_state = {}
    missing_source_keys = []
    for output_key in gpt.state_dict():
        source_key = get_source_key(output_key)
        if source_key not in checkpoint:
            missing_source_keys.append(source_key)
            continue
        converted_state[output_key] = checkpoint[source_key]
    if missing_source_keys:
        raise RuntimeError(
            "The checkpoint is missing required GPT parameters: "
            + ", ".join(missing_source_keys[:10])
        )

    gpt.load_state_dict(converted_state, strict=True, assign=True)
    del checkpoint
    del converted_state
    gpt = gpt.to(device=args.device, dtype=DTYPES[args.dtype])
    gpt.eval()

    os.makedirs(output_dir, exist_ok=True)
    gpt.save_pretrained(output_dir)

    output_config_path = os.path.join(output_dir, "config.json")
    with open(output_config_path, "r", encoding="utf-8") as file:
        output_config = json.load(file)
    if output_config.get("architectures") != ["GPT2InferenceModel"]:
        raise RuntimeError(
            "Unexpected exported architecture: "
            f"{output_config.get('architectures')}"
        )
    if output_config.get("n_positions") != max_mel_positions:
        raise RuntimeError("The exported position embedding size is incorrect")
    weight_files = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if not any(
        os.path.isfile(os.path.join(output_dir, name))
        for name in weight_files
    ):
        raise RuntimeError("Transformers did not export a model weight file")

    print(">> GPT weights restored from:", gpt_path)
    print(">> Export dtype:", args.dtype)
    print(">> GPT transformer saved to:", output_dir)

    tokenizer_path = os.path.join(
        model_dir,
        "multilingual_zh_ja_yue_char_del.tiktoken",
    )
    if not os.path.isfile(tokenizer_path):
        print(
            ">> Warning: the official IndexTTS-2.5 text tokenizer is "
            f"missing: {tokenizer_path}"
        )
    else:
        print(">> Official text tokenizer remains at:", tokenizer_path)


if __name__ == "__main__":
    main()
