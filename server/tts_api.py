"""
Chatterbox Norwegian TTS API Server
Deploy on GPU (RunPod serverless or persistent pod)
Best params: exaggeration=0.5, cfg_weight=0.5, temperature=0.8
"""

import io
import os
import time
import torch
import torchaudio as ta
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from chatterbox.tts import ChatterboxTTS
from huggingface_hub import hf_hub_download

app = FastAPI(title="EchoTrail Norwegian TTS", version="1.0")

# Global model
model = None

class TTSRequest(BaseModel):
    text: str
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    audio_prompt_path: str | None = None

@app.on_event("startup")
async def load_model():
    global model
    print("Loading Chatterbox Norwegian TTS model...")
    REPO_ID = "akhbar/chatterbox-tts-norwegian"
    for f in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
        hf_hub_download(repo_id=REPO_ID, filename=f)
    lp = hf_hub_download(repo_id=REPO_ID, filename="conds.pt")
    model = ChatterboxTTS.from_local(Path(lp).parent, device="cuda")
    print("Model loaded!")

@app.post("/tts")
async def synthesize(req: TTSRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not req.text or len(req.text) > 2000:
        raise HTTPException(status_code=400, detail="Text must be 1-2000 chars")
    
    t0 = time.time()
    wav = model.generate(
        req.text,
        exaggeration=req.exaggeration,
        cfg_weight=req.cfg_weight,
        temperature=req.temperature,
    )
    elapsed = time.time() - t0
    
    # Convert to WAV bytes
    buf = io.BytesIO()
    ta.save(buf, wav, model.sr, format="wav")
    buf.seek(0)
    
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "X-Inference-Time": f"{elapsed:.3f}",
            "X-Sample-Rate": str(model.sr),
        },
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
