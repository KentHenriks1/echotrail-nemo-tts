"""
RunPod Serverless handler for Chatterbox Norwegian TTS
Deploy as: runpod serverless endpoint
"""
import runpod
import base64
import io
import os
import torch
import torchaudio as ta
from pathlib import Path
from chatterbox.tts import ChatterboxTTS
from huggingface_hub import hf_hub_download

# Load model at cold start (once per container)
model = None

def load_model():
    global model
    if model is not None:
        return model
    print("Loading Chatterbox Norwegian TTS...")
    REPO_ID = "akhbar/chatterbox-tts-norwegian"
    for f in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
        hf_hub_download(repo_id=REPO_ID, filename=f)
    lp = hf_hub_download(repo_id=REPO_ID, filename="conds.pt")
    model = ChatterboxTTS.from_local(Path(lp).parent, device="cuda")
    print("Model loaded!")
    return model

def handler(job):
    """
    Input:
        job["input"]["text"]         - text to synthesize (required)
        job["input"]["exaggeration"] - float 0-1 (default 0.5)
        job["input"]["cfg_weight"]   - float 0-1 (default 0.5)
        job["input"]["temperature"]  - float 0-1 (default 0.8)
    Output:
        {"audio_base64": "...", "sample_rate": 22050, "duration": 2.3}
    """
    try:
        inp = job.get("input", {})
        text = inp.get("text", "")

        if not text or len(text) > 2000:
            return {"error": "text must be 1-2000 chars"}

        exaggeration = float(inp.get("exaggeration", 0.5))
        cfg_weight = float(inp.get("cfg_weight", 0.5))
        temperature = float(inp.get("temperature", 0.8))

        m = load_model()
        wav = m.generate(
            text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )

        # Convert to bytes
        buf = io.BytesIO()
        ta.save(buf, wav, m.sr, format="wav")
        audio_b64 = base64.b64encode(buf.getvalue()).decode()
        duration = wav.shape[-1] / m.sr

        return {
            "audio_base64": audio_b64,
            "sample_rate": m.sr,
            "duration": round(duration, 2),
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
