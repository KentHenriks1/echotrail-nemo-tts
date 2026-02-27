#!/bin/bash
set -e

# Install ffmpeg (required by F5-TTS for audio loading)
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1
echo "ffmpeg installed"

python3 << 'PYEOF'
import torch, os, soundfile as sf, time, numpy as np

os.makedirs('/workspace/tts_eval', exist_ok=True)

# Step 1: Create a reference audio file from scratch (sine wave, then overwrite with real speech)
# Generate a short Norwegian speech reference using soundfile directly
ref_path = '/workspace/tts_eval/ref_audio.wav'
if not os.path.exists(ref_path):
    # Create 3 seconds of silence as placeholder - F5 just needs the voice characteristics
    sr = 24000
    silence = np.zeros(sr * 3, dtype=np.float32)
    sf.write(ref_path, silence, sr)
    print(f"Created placeholder ref audio: {ref_path}")

# Step 2: Load F5-TTS
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download(repo_id='akhbar/F5_Norwegian', filename='F5_Base_Norwegian.safetensors')

from f5_tts.api import F5TTS
model = F5TTS(ckpt_file=ckpt, device='cuda')
print("F5-TTS loaded!")

# Step 3: Use chatterbox output as reference (has real Norwegian speech)
ref_audio = None
for candidate in [
    '/workspace/tts_eval/chatterbox_3.wav',
    '/workspace/tts_eval/chatterbox_1.wav',
]:
    if os.path.exists(candidate):
        ref_audio = candidate
        break

if not ref_audio:
    ref_audio = ref_path  # fallback to silence

print(f"Reference audio: {ref_audio}")

texts = [
    'Hei, jeg heter EchoTrail og jeg kan hjelpe deg med å finne veien.',
    'Det er viktig å ta vare på naturen rundt oss.',
    'Godmorgen! Hvordan har du det i dag?',
    'Norge er et vakkert land med fjorder, fjell og nordlys.',
    'Denne setningen tester om modellen håndterer lengre tekst med flere ord og naturlig prosodi.',
]

for i, text in enumerate(texts):
    print(f'[{i+1}/5] {text}')
    t0 = time.time()
    wav, sr, _ = model.infer(ref_file=ref_audio, ref_text='', gen_text=text)
    elapsed = time.time() - t0
    sf.write(f'/workspace/tts_eval/f5tts_{i+1}.wav', wav, sr)
    print(f'  {elapsed:.1f}s -> f5tts_{i+1}.wav')

print('\n=== F5-TTS DONE ===')
os.system("ls -la /workspace/tts_eval/*.wav")

# Start HTTP server for listening
print('\nStarting HTTP server on port 8888...')
print('Open: https://3d20y5h53ieb3u-8888.proxy.runpod.net/')
os.system("cd /workspace/tts_eval && python3 -m http.server 8888 &")
PYEOF
