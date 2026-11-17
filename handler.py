import base64
import io
import os
import threading
import warnings

import runpod
import torch
from PIL import Image, ImageOps
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import Qwen3ForCausalLM, BitsAndBytesConfig as TransformersBitsAndBytesConfig

MODEL = "black-forest-labs/FLUX.2-klein-9B"
REVISION = "92196c8e11f7b6cf2b7493e037d8c5345c559216"
SIZES = {(640, 640), (768, 768), (1024, 1024), (1024, 768), (768, 1024)}
Image.MAX_IMAGE_PIXELS = 20_000_000
PRECISION = os.environ.get("FLUX9B_PRECISION", "nf4").lower()


def load_pipeline():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("RunPodの環境変数HF_TOKENを設定してください。9Bの利用条件に同意したHugging FaceアカウントのReadトークンが必要です。")
    if PRECISION not in ("nf4", "bf16"):
        raise RuntimeError("FLUX9B_PRECISIONはnf4またはbf16を指定してください。")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA対応GPUのワーカーを選択してください。")
    shared = dict(revision=REVISION, token=token, torch_dtype=torch.bfloat16, use_safetensors=True, low_cpu_mem_usage=True, device_map={"": 0})
    transformer_args, encoder_args = {}, {}
    if PRECISION == "nf4":
        transformer_args["quantization_config"] = DiffusersBitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        encoder_args["quantization_config"] = TransformersBitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    # Load both large components directly onto GPU; do not retain full BF16 copies in host RAM.
    transformer = Flux2Transformer2DModel.from_pretrained(MODEL, subfolder="transformer", **shared, **transformer_args)
    text_encoder = Qwen3ForCausalLM.from_pretrained(MODEL, subfolder="text_encoder", **shared, **encoder_args)
    pipeline = Flux2KleinPipeline.from_pretrained(
        MODEL, revision=REVISION, token=token, transformer=transformer, text_encoder=text_encoder,
        torch_dtype=torch.bfloat16, use_safetensors=True, low_cpu_mem_usage=True,
    )
    # Quantized components already reside on CUDA. Move only the small, unquantized VAE.
    pipeline.vae.to(device="cuda:0", dtype=torch.bfloat16)
    return pipeline


pipe = load_pipeline()
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
    width, height = int(p.get("width", 1024)), int(p.get("height", 1024))
    if (width, height) not in SIZES:
        raise ValueError("未対応の出力サイズです。")
    prompt = str(p.get("prompt", "")).strip()
    if not 1 <= len(prompt) <= 3000:
        raise ValueError("編集指示は1〜3000文字で入力してください。")
    seed = int(p.get("seed", 42))
    if not 0 <= seed <= 2147483647:
        raise ValueError("シード値を確認してください。")
    if p.get("lora") is not None or p.get("mask_base64"):
        raise ValueError("この初期版はLoRA・マスク入力には対応していません。")
    # Keep the complete reference image; all dimensions are multiples of 32.
    image = ImageOps.pad(image, (width, height), method=Image.Resampling.LANCZOS, color="black")
    return image, prompt, width, height, seed


def handler(job):
    try:
        image, prompt, width, height, seed = parse_request(job.get("input", {}))
    except Exception as error:
        return {"error": str(error)}
    with lock, torch.inference_mode():
        result = pipe(
            image=image, prompt=prompt, width=width, height=height,
            num_inference_steps=4, guidance_scale=1.0,
            generator=torch.Generator("cpu").manual_seed(seed), num_images_per_prompt=1,
            max_sequence_length=512,
        ).images[0]
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return {
        "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "mime_type": "image/png", "mode": "flux2_klein9_i2i", "model": MODEL,
        "width": result.width, "height": result.height, "seed": seed,
        "steps": 4, "guidance": 1.0, "revision": REVISION, "precision": PRECISION,
    }


runpod.serverless.start({"handler": handler})
