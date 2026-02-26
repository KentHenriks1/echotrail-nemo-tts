#!/bin/bash
set -e

echo "=== Fix torchvision ==="
pip install --force-reinstall torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3

echo ""
echo "=== Install F5-TTS ==="
pip install f5-tts 2>&1 | tail -5

echo ""
echo "=== MODEL 1: F5-TTS Norwegian ==="
python3 << 'PYEOF'
import torch, os, soundfile as sf, glob, time
os.makedirs('/workspace/tts_eval', exist_ok=True)

try:
    from f5_tts.api import F5TTS
    print("Loading F5-TTS Norwegian model...")
    model = F5TTS(model_type='F5-TTS', ckpt_file='hf://akhbar/F5_Norwegian', device='cuda')
    print("Model loaded!")

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
    else:
        print("ERROR: No reference audio found in /workspace/echotrail-nemo-tts/data/wavs/")
    del model
    torch.cuda.empty_cache()
    print('=== F5-TTS DONE ===')
except Exception as e:
    print(f'F5-TTS ERROR: {e}')
    import traceback; traceback.print_exc()
PYEOF

echo ""
echo "=== Install Chatterbox ==="
pip install chatterbox-tts 2>&1 | tail -5

echo ""
echo "=== MODEL 2: Chatterbox Norwegian ==="
python3 << 'PYEOF'
import torch, os, time
os.makedirs('/workspace/tts_eval', exist_ok=True)

try:
    from pathlib import Path
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS
    from huggingface_hub import hf_hub_download

    print("Downloading Chatterbox Norwegian model...")
    REPO_ID = 'akhbar/chatterbox-tts-norwegian'
    for f in ['ve.safetensors', 't3_cfg.safetensors', 's3gen.safetensors', 'tokenizer.json', 'conds.pt']:
        lp = hf_hub_download(repo_id=REPO_ID, filename=f)

    print("Loading model...")
    model = ChatterboxTTS.from_local(Path(lp).parent, device='cuda')
    print("Model loaded!")

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
        wav = model.generate(text, exaggeration=1.0, cfg_weight=0.5, temperature=0.4)
        elapsed = time.time() - t0
        ta.save(f'/workspace/tts_eval/chatterbox_{i+1}.wav', wav, model.sr)
        print(f'  Time: {elapsed:.3f}s  Saved: chatterbox_{i+1}.wav\n')

    del model
    torch.cuda.empty_cache()
    print('=== Chatterbox DONE ===')
except Exception as e:
    print(f'Chatterbox ERROR: {e}')
    import traceback; traceback.print_exc()
PYEOF

echo ""
echo "=========================================="
echo "RESULTS"
echo "=========================================="
ls -la /workspace/tts_eval/*.wav 2>/dev/null || echo "No WAV files generated"
echo ""
echo "To listen, download files from: /workspace/tts_eval/"
echo "Or start a simple HTTP server:"
echo "  cd /workspace/tts_eval && python3 -m http.server 8888"
