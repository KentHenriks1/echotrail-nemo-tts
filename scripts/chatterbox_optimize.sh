#!/bin/bash
set -e
echo "=== Chatterbox Norwegian Parameter Optimization ==="
echo "Testing different ref audio, exaggeration, cfg_weight, temperature"

python3 << 'PYEOF'
import torch, os, time, base64
import torchaudio as ta
from pathlib import Path
from chatterbox.tts import ChatterboxTTS
from huggingface_hub import hf_hub_download

os.makedirs('/workspace/tts_eval/optimize', exist_ok=True)

# Load model
REPO_ID = 'akhbar/chatterbox-tts-norwegian'
for f in ['ve.safetensors', 't3_cfg.safetensors', 's3gen.safetensors', 'tokenizer.json', 'conds.pt']:
    lp = hf_hub_download(repo_id=REPO_ID, filename=f)
model = ChatterboxTTS.from_local(Path(lp).parent, device='cuda')
print("Model loaded!")

# Test sentence
test_text = "Hei, jeg heter EchoTrail og jeg kan hjelpe deg med å finne veien. Norge er et vakkert land med fjorder og fjell."

# Parameter grid
configs = [
    {"exaggeration": 0.3, "cfg_weight": 0.5, "temperature": 0.4, "label": "low_exag"},
    {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.4, "label": "med_exag"},
    {"exaggeration": 1.0, "cfg_weight": 0.5, "temperature": 0.4, "label": "high_exag"},
    {"exaggeration": 0.5, "cfg_weight": 0.3, "temperature": 0.4, "label": "low_cfg"},
    {"exaggeration": 0.5, "cfg_weight": 0.7, "temperature": 0.4, "label": "high_cfg"},
    {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.2, "label": "low_temp"},
    {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.6, "label": "med_temp"},
    {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8, "label": "high_temp"},
    {"exaggeration": 0.3, "cfg_weight": 0.3, "temperature": 0.2, "label": "conservative"},
    {"exaggeration": 0.7, "cfg_weight": 0.5, "temperature": 0.5, "label": "balanced"},
]

# Also test with a voice cloning reference
# Use a clean Norwegian audio sample if available
ref_audio = None
import glob
nst_wavs = sorted(glob.glob('/workspace/echotrail-nemo-tts/data/wavs/*.wav'))
if nst_wavs:
    # Pick a medium-length sample
    for w in nst_wavs[100:110]:
        info = ta.info(w)
        if 3.0 < info.num_frames / info.sample_rate < 8.0:
            ref_audio = w
            break
    if not ref_audio:
        ref_audio = nst_wavs[100]

print(f"Reference audio for voice cloning: {ref_audio}")

files_generated = []

for cfg in configs:
    label = cfg.pop("label")
    print(f"\n--- {label}: exag={cfg['exaggeration']}, cfg={cfg['cfg_weight']}, temp={cfg['temperature']} ---")
    
    t0 = time.time()
    wav = model.generate(test_text, **cfg)
    elapsed = time.time() - t0
    
    fname = f"cb_{label}.wav"
    fpath = f"/workspace/tts_eval/optimize/{fname}"
    ta.save(fpath, wav, model.sr)
    files_generated.append(fname)
    print(f"  {elapsed:.1f}s -> {fname}")

# Test with voice cloning if we have ref audio
if ref_audio:
    print(f"\n--- Voice cloning with ref: {ref_audio} ---")
    for exag in [0.3, 0.5, 0.7]:
        t0 = time.time()
        wav = model.generate(test_text, audio_prompt_path=ref_audio, exaggeration=exag, cfg_weight=0.5, temperature=0.4)
        elapsed = time.time() - t0
        fname = f"cb_clone_exag{exag}.wav"
        fpath = f"/workspace/tts_eval/optimize/{fname}"
        ta.save(fpath, wav, model.sr)
        files_generated.append(fname)
        print(f"  exag={exag}: {elapsed:.1f}s -> {fname}")

# Build HTML with all variants
print("\n=== Building comparison HTML ===")
html = '<html><body><h1>Chatterbox Norwegian - Parameter Optimization</h1>'
html += f'<p>Test: "{test_text}"</p>'
html += '<table border="1" cellpadding="8"><tr><th>Variant</th><th>Audio</th></tr>'

for fname in files_generated:
    fpath = f"/workspace/tts_eval/optimize/{fname}"
    with open(fpath, 'rb') as fp:
        b = base64.b64encode(fp.read()).decode()
    html += f'<tr><td><b>{fname}</b></td><td><audio controls src="data:audio/wav;base64,{b}"></audio></td></tr>'

html += '</table></body></html>'

with open('/workspace/tts_eval/optimize/compare.html', 'w') as fp:
    fp.write(html)
print(f"HTML: {len(html)} bytes, {len(files_generated)} variants")

# Copy to repo and push
import shutil
shutil.copy('/workspace/tts_eval/optimize/compare.html', '/workspace/echotrail-nemo-tts/compare.html')
os.chdir('/workspace/echotrail-nemo-tts')
os.system('git add compare.html && git commit -m "Chatterbox parameter optimization comparison" && git push')

print("\n=== DONE ===")
PYEOF
