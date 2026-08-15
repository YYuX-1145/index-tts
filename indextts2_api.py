import os
import sys
import traceback

now_dir = os.getcwd()

import argparse
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from io import BytesIO

from pydantic import BaseModel

"""
import torchaudio
### monkey patch
_original_torchaudio_save = torchaudio.save
def patched_save(uri, src, sample_rate, format=None, **kwargs):
    if format is None:
        format = 'wav'
    return _original_torchaudio_save(uri, src, sample_rate, format=format, **kwargs)
torchaudio.save = patched_save
###
"""

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--version", type=str, default="2.5", choices=["2", "2.5"], help="Model version to use")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--use_deepspeed", action="store_true", default=False, help="Use Deepspeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use cuda kernel for inference if available")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
parser.add_argument("--no_qwen_emo", action="store_true", default=False, help="Disable Qwen_emotion, which can save about 2GB VRAM, but text emotion prompt will be no longer available.")
args = parser.parse_args()

IS_V25 = args.version == "2.5"
if IS_V25:
    from indextts.infer_vllm_v2_5 import IndexTTS2
else:
    from indextts.infer_vllm_v2 import IndexTTS2

# device = args.device
port = args.port
host = args.bind_addr
argv = sys.argv


APP = FastAPI()


class TTS_Request(BaseModel):
    text: str = None
    emo_text: str = None
    ref_audio_path: str = None
    emo_ref_audio_path: str = None
    top_k: int = 30
    top_p: float = 0.8
    temperature: float = 0.8
    emo_alpha: float = 0.7
    emo_vec: list = []
    normalize_emo_vec: bool = False
    speed_factor: float = 1.0
    seed: int = -1
    parallel_infer: bool = True
    repetition_penalty: float = 10
    lang: str = "ZH"
    duration_factor: float | None = None
    text_normalization: bool = True


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def check_params(req: dict):
    text: str = req.get("text", "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    return None


async def tts_handle(req: dict):
    check_res = check_params(req)
    if check_res is not None:
        return check_res
    try:
        emo_text = req["emo_text"]
        use_emo_text = bool(req["emo_text"])
        if emo_text == 'auto':
            emo_text = None
            use_emo_text = True
        emo_vec = req["emo_vec"]
        if len(emo_vec) == 0:
            emo_vec = None
        else:
            use_emo_text = False
            emo_text = None
            if req["normalize_emo_vec"]:
                emo_vec = tts_pipeline.normalize_emo_vec(emo_vec)
        infer_kwargs = dict(
            spk_audio_prompt=req["ref_audio_path"],
            emo_audio_prompt=req["emo_ref_audio_path"] if req["emo_ref_audio_path"] else None,
            text=req["text"],
            emo_text=emo_text,
            use_emo_text=use_emo_text,
            emo_alpha=req["emo_alpha"],
            emo_vector=emo_vec,
            top_p=req["top_p"],
            top_k=req["top_k"],
            temperature=req["temperature"],
            repetition_penalty=req["repetition_penalty"],
            output_path=None,
        )
        if IS_V25:
            duration_factor = req.get("duration_factor")
            if duration_factor is None:
                speed_factor = float(req.get("speed_factor", 1.0))
                if speed_factor <= 0:
                    raise ValueError("speed_factor must be greater than zero")
                duration_factor = 1.0 / speed_factor
            infer_kwargs.update(
                lang=req.get("lang", "ZH"),
                duration_factor=duration_factor,
                text_normalization=req.get("text_normalization", True),
            )
        sampling_rate, wav_data = await tts_pipeline.infer(**infer_kwargs)
        return Response(pack_wav(BytesIO(), wav_data, sampling_rate).getvalue(), media_type=f"audio/wav")
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": str(e)})


@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)


@APP.get("/tts")
async def tts_get_endpoint(
    text: str = None,
    emo_text: str = None,
    ref_audio_path: str = None,
    emo_ref_audio_path: str = None,
    top_k: int = 30,
    top_p: float = 0.8,
    temperature: float = 0.8,
    emo_alpha: float = 0.7,
    normalize_emo_vec: bool = False,
    speed_factor: float = 1.0,
    seed: int = -1,
    parallel_infer: bool = True,
    repetition_penalty: float = 10,
    lang: str = "ZH",
    duration_factor: float | None = None,
    text_normalization: bool = True,
):
    req = {
        "text": text,
        "emo_text": emo_text,
        "ref_audio_path": ref_audio_path,
        "emo_ref_audio_path": emo_ref_audio_path,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "emo_alpha": float(emo_alpha),
        "emo_vec": [],
        "normalize_emo_vec": normalize_emo_vec,
        "speed_factor": float(speed_factor),
        "seed": seed,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty),
        "lang": lang,
        "duration_factor": (
            float(duration_factor) if duration_factor is not None else None
        ),
        "text_normalization": text_normalization,
    }
    return await tts_handle(req)


@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    return await tts_handle(req)


# @APP.get("/set_gpt_weights")
# async def set_gpt_weights(weights_path: str = None):
#     return JSONResponse(status_code=200, content={"message": "index不需要切换模型"})


# @APP.get("/set_sovits_weights")
# async def set_sovits_weights(weights_path: str = None):
#     return JSONResponse(status_code=200, content={"message": "index不需要切换模型"})


if __name__ == "__main__":
    init_kwargs = dict(
        model_dir=args.model_dir,
        cfg_path=os.path.join(args.model_dir, "config.yaml"),
        gpu_memory_utilization=args.gpu_memory_utilization,
        use_cuda_kernel=args.cuda_kernel,
        use_qwen_emo=not args.no_qwen_emo,
    )
    if IS_V25:
        import torch

        use_bf16 = args.fp16 and torch.cuda.is_bf16_supported()
        if args.fp16 and not use_bf16:
            print(">> BF16 is not supported; using full precision for V2.5")
        init_kwargs["use_bf16"] = use_bf16
    else:
        init_kwargs["is_fp16"] = args.fp16
    tts_pipeline = IndexTTS2(**init_kwargs)
    try:
        if host == "None":
            host = None
        uvicorn.run(app=APP, host=host, port=port)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
