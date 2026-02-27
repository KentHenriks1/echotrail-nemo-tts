#!/bin/bash
set -e

echo "=== F5-TTS Norwegian eval ==="
python3 << 'PYEOF'
import torch, os, soundfile as sf, glob, time
os.makedirs('/workspace/tts_eval', exist_ok=True)

from huggingface_hub import hf_hub_download
print("Downloading F5_Base_Norwegian.safetensors (1.35 GB)...")
ckpt = hf_hub_download(repo_id='akhbar/F5_Norwegian', filename='F5_Base_Norwegian.safetensors')
print(f"Checkpoint: {ckpt}")

from f5_tts.api import F5TTS
import inspect
sig = inspect.signature(F5TTS.__init__)
print(f"F5TTS params: {sig}")

# Try loading with correct checkpoint path
try:
    model = F5TTS(ckpt_file=ckpt, device='cuda')
    print("Model loaded!")
except Exception as e:
    print(f"Direct load failed: {e}")
    # Try with vocab file from base model
    try:
        vocab_path = hf_hub_download(repo_id='SWivid/F5-TTS', filename='data/Emilia_ZH_EN_pinyin/vocab.txt')
        model = F5TTS(ckpt_file=ckpt, vocab_file=vocab_path, device='cuda')
        print("Model loaded with vocab!")
    except Exception as e2:
        print(f"With vocab failed: {e2}")
        import traceback; traceback.print_exc()
        exit(1)

ref_wavs = sorted(glob.glob('/workspace/echotrail-nemo-tts/data/wavs/*.wav'))
ref_audio = ref_wavs[0] if ref_wavs else None
print(f"Reference audio: {ref_audio}")

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

print("\nAll files:")
os.system("ls -la /workspace/tts_eval/")
PYEOF
