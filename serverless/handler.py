"""
RunPod Serverless handler for Chatterbox Norwegian TTS
Output format matches EchoTrail tts-selfhosted.mjs expectations:
  { audio, duration_seconds, model_version, sample_rate }
"""
import runpod
import base64
import io
import time
import torch
import torchaudio as ta
from pathlib import Path
from chatterbox.tts import ChatterboxTTS
from huggingface_hub import hf_hub_download

MODEL_VERSION = "chatterbox-norwegian-v1"
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
    print(f"Model loaded! ({MODEL_VERSION})")
    return model

def handler(job):
    """
    Expected input (from tts-selfhosted.mjs):
        job["input"]["text"]         - text to synthesize
        job["input"]["voice_id"]     - ignored (single voice model)
        job["input"]["language"]     - ignored (Norwegian only)
        job["input"]["speed"]        - ignored for now
        job["input"]["output_format"]- ignored (returns WAV, client handles)

    Output (consumed by tts-selfhosted.mjs):
        audio           - base64 WAV audio
        duration_seconds - float
        model_version   - string
        sample_rate     - int
    """
    try:
        inp = job.get("input", {})
        text = inp.get("text", "").strip()

        if not text:
            return {"error": "text is required"}
        if len(text) > 2000:
            return {"error": "text exceeds 2000 char limit"}

        exaggeration = float(inp.get("exaggeration", 0.5))
        cfg_weight = float(inp.get("cfg_weight", 0.5))
        temperature = float(inp.get("temperature", 0.8))

        m = load_model()
        t0 = time.time()
        wav = m.generate(
            text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )
        elapsed = time.time() - t0

        buf = io.BytesIO()
        ta.save(buf, wav, m.sr, format="wav")
        audio_b64 = base64.b64encode(buf.getvalue()).decode()
        duration = round(wav.shape[-1] / m.sr, 2)

        print(f"Generated {duration}s audio in {elapsed:.2f}s for: {text[:80]}")

        return {
            "audio": audio_b64,
            "duration_seconds": duration,
            "model_version": MODEL_VERSION,
            "sample_rate": m.sr,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
