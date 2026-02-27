#!/bin/bash
set -e
echo "=== XTTS v2 Norwegian Evaluation ==="

# Install dependencies
pip install coqui-tts -q 2>&1 | tail -3
apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null 2>&1

python3 << 'PYEOF'
import torch, os, time, base64
os.makedirs('/workspace/tts_eval', exist_ok=True)

from TTS.api import TTS

# XTTS v2 supports Norwegian out of the box
print("Loading XTTS v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
print("Model loaded!")

# We need a reference audio for voice cloning
# Generate a short reference first using a built-in speaker
# Or download a clean Norwegian sample
ref_audio = "/workspace/tts_eval/ref_norwegian.wav"
if not os.path.exists(ref_audio):
    # Use XTTS to generate a reference with default voice first
    print("Generating reference audio...")
    tts.tts_to_file(
        text="Hei, dette er en test av norsk tale.",
        language="no",
        file_path=ref_audio,
    )
    print(f"Reference: {ref_audio}")

texts = [
    'Hei, jeg heter EchoTrail og jeg kan hjelpe deg med å finne veien.',
    'Det er viktig å ta vare på naturen rundt oss.',
    'Godmorgen! Hvordan har du det i dag?',
    'Norge er et vakkert land med fjorder, fjell og nordlys.',
    'Denne setningen tester om modellen håndterer lengre tekst med flere ord og naturlig prosodi.',
]

# Test 1: Default voice (no cloning)
print("\n=== XTTS v2 - Default voice ===")
for i, text in enumerate(texts):
    print(f'[{i+1}/5] {text}')
    t0 = time.time()
    tts.tts_to_file(text=text, language="no", file_path=f"/workspace/tts_eval/xtts_default_{i+1}.wav")
    print(f'  {time.time()-t0:.1f}s -> xtts_default_{i+1}.wav')

# Test 2: With voice cloning (using ref audio)
print("\n=== XTTS v2 - Voice cloning ===")
for i, text in enumerate(texts):
    print(f'[{i+1}/5] {text}')
    t0 = time.time()
    tts.tts_to_file(
        text=text,
        language="no",
        speaker_wav=ref_audio,
        file_path=f"/workspace/tts_eval/xtts_clone_{i+1}.wav",
    )
    print(f'  {time.time()-t0:.1f}s -> xtts_clone_{i+1}.wav')

# Build comparison HTML with XTTS + previous Chatterbox results
print("\n=== Building comparison HTML ===")
wavdir = '/workspace/tts_eval'
files = sorted([f for f in os.listdir(wavdir) if f.endswith('.wav') and f != 'ref_audio.wav' and f != 'ref_norwegian.wav'])
html = '<html><body><h1>Norwegian TTS: XTTS v2 vs Chatterbox</h1><table border="1" cellpadding="8"><tr><th>File</th><th>Audio</th></tr>'
for f in files:
    with open(os.path.join(wavdir, f), 'rb') as fp:
        b = base64.b64encode(fp.read()).decode()
    html += f'<tr><td><b>{f}</b></td><td><audio controls src="data:audio/wav;base64,{b}"></audio></td></tr>'
html += '</table></body></html>'
with open(f'{wavdir}/xtts_compare.html', 'w') as fp:
    fp.write(html)
print(f'HTML: {len(html)} bytes, {len(files)} files')

# Push to GitHub
import shutil
shutil.copy(f'{wavdir}/xtts_compare.html', '/workspace/echotrail-nemo-tts/xtts_compare.html')
os.chdir('/workspace/echotrail-nemo-tts')
os.system('git add xtts_compare.html && git commit -m "XTTS v2 Norwegian evaluation" && git push')
print("\n=== DONE ===")
PYEOF
