import base64
import io
import math
import os
import re
import traceback
import importlib.metadata
import threading
import warnings
from pathlib import PurePosixPath

import runpod
torch = None
from PIL import Image, ImageOps

MODEL = "black-forest-labs/FLUX.2-klein-9B"
REVISION = "92196c8e11f7b6cf2b7493e037d8c5345c559216"
SIZES = {(640, 640), (768, 768), (1024, 1024), (1024, 768), (768, 1024)}
Image.MAX_IMAGE_PIXELS = 20_000_000
PRECISION = os.environ.get("FLUX9B_PRECISION", "nf4").strip().lower()
WORKER_VERSION = "klein9-lora-v3-sizing"
phase = "registered"
load_error = None

loaded_lora = None
lora_failed = False


def safe_text(value):
    text = str(value)
    for key in ("HF_TOKEN", "RUNPOD_API_KEY"):
        secret = os.environ.get(key, "").strip()
        if secret:
            text = text.replace(secret, "[redacted]")
    text = re.sub(r"hf_[A-Za-z0-9_]+", "[redacted-hf-token]", text)
    text = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", text, flags=re.I)
    return text[:3000]


def failure(error, failed_phase):
    return {
        "error": safe_text(f"{failed_phase}: {type(error).__name__}: {error}"),
        "diagnostic": {
            "worker_version": WORKER_VERSION,
            "phase": failed_phase,
            "error_type": type(error).__name__,
            "traceback": safe_text(traceback.format_exc(limit=6)),
        },
    }


def runtime_imports():
    global torch, Flux2KleinPipeline, Flux2Transformer2DModel, DiffusersBitsAndBytesConfig
    global Qwen3ForCausalLM, TransformersBitsAndBytesConfig
    import torch
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    from transformers import Qwen3ForCausalLM, BitsAndBytesConfig as TransformersBitsAndBytesConfig
    import peft


def diagnose():
    global phase
    report = {
        "worker_version": WORKER_VERSION,
        "precision": PRECISION,
        "hf_token_configured": bool(os.environ.get("HF_TOKEN", "").strip()),
        "model_weights_downloaded": False,
    }
    try:
        phase = "configuration"
        token = os.environ.get("HF_TOKEN", "").strip()
        if not token:
            raise RuntimeError("HF_TOKENが環境変数にありません。")
        if not token.startswith("hf_"):
            raise RuntimeError("HF_TOKENはHugging Faceのhf_で始まるトークンを設定してください。")
        if PRECISION not in ("nf4", "bf16"):
            raise RuntimeError("FLUX9B_PRECISIONはnf4またはbf16にしてください。")
        phase = "dependency_imports"
        runtime_imports()
        if PRECISION == "nf4":
            import bitsandbytes
        report["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("torch", "diffusers", "transformers", "accelerate", "bitsandbytes", "peft")
        }
        phase = "gpu"
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPUを認識できません。GPUワーカーの選択を確認してください。")
        report["gpu"] = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        report["vram_free_gib"] = round(free / 1024**3, 2)
        report["vram_total_gib"] = round(total / 1024**3, 2)
        phase = "hugging_face_access"
        from huggingface_hub import hf_hub_url, get_hf_file_metadata
        get_hf_file_metadata(hf_hub_url(MODEL, "model_index.json", revision=REVISION), token=token, timeout=20)
        report["model_access"] = "ok"
        report["ok"] = True
        phase = "diagnostic_complete"
        return {"diagnostic": report}
    except Exception as error:
        output = failure(error, phase)
        output["diagnostic"].update(report)
        output["diagnostic"]["ok"] = False
        return output


def load_pipeline():
    global phase
    phase = "configuration"
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("RunPodの環境変数HF_TOKENを設定してください。9Bの利用条件に同意したHugging FaceアカウントのReadトークンが必要です。")
    if PRECISION not in ("nf4", "bf16"):
        raise RuntimeError("FLUX9B_PRECISIONはnf4またはbf16を指定してください。")
    phase = "dependency_imports"
    runtime_imports()
    phase = "gpu"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA対応GPUのワーカーを選択してください。")
    shared = dict(
        revision=REVISION,
        token=token,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    transformer_args, encoder_args = {}, {}
    if PRECISION == "nf4":
        transformer_args["quantization_config"] = DiffusersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        encoder_args["quantization_config"] = TransformersBitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    phase = "transformer_load"
    transformer = Flux2Transformer2DModel.from_pretrained(MODEL, subfolder="transformer", **shared, **transformer_args)
    phase = "text_encoder_load"
    text_encoder = Qwen3ForCausalLM.from_pretrained(MODEL, subfolder="text_encoder", **shared, **encoder_args)
    phase = "pipeline_assembly"
    pipeline = Flux2KleinPipeline.from_pretrained(
        MODEL,
        revision=REVISION,
        token=token,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    pipeline.vae.to(device="cuda:0", dtype=torch.bfloat16)
    phase = "ready"
    return pipeline


pipe = None
lock = threading.Lock()


def parse_request(p):
    if p.get("mode") != "flux2_klein9_i2i":
        raise ValueError("FLUX.2 klein 9B用アプリから送信してください。")

    encoded = p.get("image_base64")
    if not isinstance(encoded, str) or not 1 <= len(encoded) <= 4_000_000:
        raise ValueError("元画像が必要です。送信画像は3MB以下にしてください。")

    raw = base64.b64decode(encoded, validate=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("JPEG・PNG・WebPを選んでください。")
            image = ImageOps.exif_transpose(image).convert("RGB")

    width = int(p.get("width", 1024))
    height = int(p.get("height", 1024))
    if any(n < 256 or n > 2048 or n % 32 for n in (width, height)) or width * height > 4_194_304:
        raise ValueError("未対応の出力サイズです。")

    prompt = str(p.get("prompt", "")).strip()
    if not 1 <= len(prompt) <= 3000:
        raise ValueError("編集指示は1〜3000文字で入力してください。")

    seed = int(p.get("seed", 42))
    if not 0 <= seed <= 2147483647:
        raise ValueError("シード値を確認してください。")

    if p.get("mask_base64"):
        raise ValueError("このバージョンはマスク入力には対応していません。")

    lora = p.get("lora")
    if lora is not None:
        if not isinstance(lora, dict):
            raise ValueError("LoRA設定を確認してください。")
        repo = lora.get("repo", "")
        filename = lora.get("filename", "")
        if not isinstance(repo, str) or not isinstance(filename, str):
            raise ValueError("LoRAのrepoとfilenameは文字列で指定してください。")
        repo, filename = repo.strip(), filename.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("LoRAのrepoは公開Hugging FaceリポジトリIDを指定してください。例: author/repository")
        if (len(filename) > 200 or not re.fullmatch(r"[A-Za-z0-9_./ -]+\.safetensors", filename)
                or filename.startswith("/") or ".." in PurePosixPath(filename).parts):
            raise ValueError("LoRAのfilenameはリポジトリ内の.safetensorsファイルを指定してください。")
        raw_scale = lora.get("scale", 0.85)
        if isinstance(raw_scale, bool):
            raise ValueError("LoRAのscaleは数値で指定してください。")
        try:
            scale = float(raw_scale)
        except (TypeError, ValueError):
            raise ValueError("LoRAのscaleは数値で指定してください。") from None
        if not math.isfinite(scale) or not 0.05 <= scale <= 1.5:
            raise ValueError("LoRAのscaleは0.05〜1.5の範囲で指定してください。")
        lora = {"repo": repo, "filename": filename, "scale": scale}

    image = ImageOps.pad(image, (width, height), method=Image.Resampling.LANCZOS, color="black")
    return image, prompt, width, height, seed, lora


def unload_current_lora():
    global loaded_lora, lora_failed
    if loaded_lora is None:
        return
    try:
        pipe.unload_lora_weights()
    except Exception as error:
        lora_failed = True
        raise RuntimeError("LoRAを解除できませんでした。誤った状態での生成を防ぐため停止しました。Workerを再起動してください。") from error
    loaded_lora = None


def configure_lora(lora):
    global loaded_lora, lora_failed
    if lora_failed:
        raise RuntimeError("LoRA変更に以前失敗した状態です。Workerを再起動してから再実行してください。")
    if lora is None:
        unload_current_lora()
        return None

    from huggingface_hub import HfApi, hf_hub_download
    from safetensors import safe_open
    identity = (lora["repo"], lora["filename"])
    mutating = False
    try:
        if loaded_lora is None or loaded_lora[:2] != identity:
            info = HfApi().model_info(lora["repo"], files_metadata=True, token=False)
            file_info = next((item for item in info.siblings if item.rfilename == lora["filename"]), None)
            if file_info is None or file_info.size is None or file_info.size > 1024 * 1024 * 1024:
                raise ValueError("LoRAファイルが見つからないか、1GBを超えています。")
            path = hf_hub_download(lora["repo"], filename=lora["filename"], revision=info.sha, token=False)
            # Header-only validation avoids holding an extra full copy of the LoRA tensors in host RAM.
            with safe_open(path, framework="pt", device="cpu") as weights:
                if not weights.keys():
                    raise ValueError("LoRAファイルが空です。")
            mutating = True
            unload_current_lora()
            # Let the native loader retain safetensors metadata and handle supported Flux2 LoRA formats.
            pipe.load_lora_weights(path, adapter_name="style", use_safetensors=True)
            loaded_lora = (*identity, info.sha)
        mutating = True
        pipe.enable_lora()
        pipe.set_adapters(["style"], adapter_weights=[lora["scale"]])
        return {**lora, "revision": loaded_lora[2]}
    except Exception as error:
        if mutating:
            lora_failed = True
        raise RuntimeError("LoRAの変更に失敗しました: " + safe_text(error)) from error


def handler(job):
    global pipe, load_error, phase
    p = job.get("input", {})

    if p.get("diagnose") is True:
        with lock:
            return diagnose()

    try:
        image, prompt, width, height, seed, lora = parse_request(p)
    except Exception as error:
        return failure(error, "input_validation")

    with lock:
        if load_error is not None:
            return load_error
        try:
            if pipe is None:
                pipe = load_pipeline()
        except Exception as error:
            load_error = failure(error, phase)
            print(load_error["error"], flush=True)
            return load_error

        try:
            phase = "lora"
            used_lora = configure_lora(lora)

            phase = "inference"
            with torch.inference_mode():
                result = pipe(
                    image=image,
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=4,
                    guidance_scale=1.0,
                    generator=torch.Generator("cpu").manual_seed(seed),
                    num_images_per_prompt=1,
                    max_sequence_length=512,
                ).images[0]

            buffer = io.BytesIO()
            result.save(buffer, format="PNG")
            phase = "ready"

            return {
                "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "mime_type": "image/png",
                "mode": "flux2_klein9_i2i",
                "model": MODEL,
                "width": result.width,
                "height": result.height,
                "seed": seed,
                "steps": 4,
                "guidance": 1.0,
                "revision": REVISION,
                "precision": PRECISION,
                "worker_version": WORKER_VERSION,
                "lora": used_lora,
            }
        except Exception as error:
            return failure(error, phase)


print("FRAME klein9-lora-v3-sizing: handler registered; model loads on generation request", flush=True)
runpod.serverless.start({"handler": handler})
