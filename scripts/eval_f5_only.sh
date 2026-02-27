#!/bin/bash
set -e

echo "=== F5-TTS Norwegian eval ==="
echo "Finding reference WAV files..."
find /workspace -name "*.wav" -type f 2>/dev/null | head -10

python3 << 'PYEOF'
import torch, os, soundfile as sf, glob, time
os.makedirs('/workspace/tts_eval', exist_ok=True)

from huggingface_hub import hf_hub_download
ckpt = hf_hub_download(repo_id='akhbar/F5_Norwegian', filename='F5_Base_Norwegian.safetensors')

from f5_tts.api import F5TTS
model = F5TTS(ckpt_file=ckpt, device='cuda')
print("Model loaded!")

# Search for WAV files in multiple locations
ref_audio = None
search_paths = [
    '/workspace/echotrail-nemo-tts/data/wavs/*.wav',
    '/workspace/echotrail-nemo-tts/data/**/*.wav',
    '/workspace/data/**/*.wav',
    '/workspace/**/*.wav',
    '/workspace/tts_eval/chatterbox_3.wav',  # Use chatterbox output as fallback
]
for pattern in search_paths:
    wavs = sorted(glob.glob(pattern, recursive=True))
    if wavs:
        ref_audio = wavs[0]
        print(f"Found ref audio: {ref_audio} (from pattern: {pattern})")
        break

if not ref_audio:
    # Generate a simple reference audio
    print("No WAV found, generating reference with chatterbox output...")
    if os.path.exists('/workspace/tts_eval/chatterbox_3.wav'):
        ref_audio = '/workspace/tts_eval/chatterbox_3.wav'
        print(f"Using chatterbox output as reference: {ref_audio}")

if ref_audio:
    texts = [
        'Hei, jeg heter EchoTrail og jeg kan hjelpe deg med å finne veien.',
        'Det er viktig å ta vare på naturen rundt oss.',
        'Godmorgen! Hvordan har du det i dag?',
        'Norge er et vakkert land med fjorder, fjell og nordlys.',
        'Denne setningen tester om modellen håndterer lengre tekst med flere ord og naturlig prosodi.',
    ]
    for i, text in enumerate(texts):
        print(f'[{i+1}/5] Text: {text}')
        t0 = time.time()
        wav, sr, _ = model.infer(ref_file=ref_audio, ref_text='', gen_text=text)
        elapsed = time.time() - t0
        sf.write(f'/workspace/tts_eval/f5tts_{i+1}.wav', wav, sr)
        print(f'  Time: {elapsed:.3f}s  Saved: f5tts_{i+1}.wav\n')
    print('=== F5-TTS DONE ===')
else:
    print("ERROR: No reference audio found anywhere!")

print("\nAll files:")
os.system("ls -la /workspace/tts_eval/")
PYEOF
