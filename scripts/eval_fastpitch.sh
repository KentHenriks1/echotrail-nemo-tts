#!/bin/bash
set -e
echo "=== FastPitch Norwegian Inference ==="
python3 << 'PYEOF'
import torch, os, soundfile as sf, time, glob, tempfile, tarfile, json
import numpy as np

os.makedirs('/workspace/tts_eval', exist_ok=True)
nemo_path = '/workspace/echotrail-nemo-tts/models/norwegian_fastpitch.nemo'

# Extract .nemo to get weights
tmpdir = tempfile.mkdtemp()
with tarfile.open(nemo_path, 'r:') as tar:
    tar.extractall(tmpdir)

ckpt_files = glob.glob(f'{tmpdir}/**/model_weights.ckpt', recursive=True)
if not ckpt_files:
    ckpt_files = glob.glob(f'{tmpdir}/**/*.ckpt', recursive=True)
saved_state = torch.load(ckpt_files[0], map_location='cpu')
n_tokens = saved_state['fastpitch.encoder.word_emb.weight'].shape[0]
emb_dim = saved_state['fastpitch.encoder.word_emb.weight'].shape[1]
print(f'Embedding: {n_tokens} x {emb_dim}')

# Build tokenizer from training manifest
manifest_path = '/workspace/echotrail-nemo-tts/data/manifests/norwegian_train_ipa.json'
chars = set()
with open(manifest_path) as f:
    for line in f:
        for ch in json.loads(line).get('text', ''):
            chars.add(ch)
vocab = ['<pad>'] + sorted(chars)
char2id = {ch: i for i, ch in enumerate(vocab)}
print(f'Vocab: {len(vocab)} tokens (match: {len(vocab)==n_tokens})')

# Load English FastPitch, resize embedding, load our weights
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
fp = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
fp.fastpitch.encoder.word_emb = torch.nn.Embedding(n_tokens, emb_dim, padding_idx=0)
fp.load_state_dict(saved_state, strict=False)
fp.eval().cuda()

# Load HiFi-GAN
hg = None
for name in ["tts_en_hifigan", "tts_hifigan"]:
    try:
        hg = HifiGanModel.from_pretrained(model_name=name)
        print(f'HiFi-GAN: {name}')
        break
    except:
        continue
hg.eval().cuda()

from phonemizer import phonemize

texts = [
    'Hei, jeg heter EchoTrail og jeg kan hjelpe deg med aa finne veien.',
    'Det er viktig aa ta vare paa naturen rundt oss.',
    'Godmorgen! Hvordan har du det i dag?',
    'Norge er et vakkert land med fjorder, fjell og nordlys.',
    'Denne setningen tester om modellen haandterer lengre tekst med flere ord og naturlig prosodi.',
]

for i, text in enumerate(texts):
    ipa = phonemize(text, language='nb', backend='espeak', strip=True,
                    preserve_punctuation=True, language_switch='remove-flags')
    print(f'\n[{i+1}/5] {text}')
    print(f'  IPA: {ipa}')

    token_ids = [char2id[ch] for ch in ipa if ch in char2id]
    print(f'  Tokens: {len(token_ids)}, max={max(token_ids)}')

    t0 = time.time()
    with torch.no_grad():
        tokens_t = torch.tensor([token_ids], dtype=torch.long).cuda()
        token_len = torch.tensor([len(token_ids)], dtype=torch.long).cuda()

        # Use generate_spectrogram which is the high-level model API
        spec = fp.generate_spectrogram(tokens=tokens_t)
        audio = hg.convert_spectrogram_to_audio(spec=spec).squeeze().cpu().numpy()

    elapsed = time.time() - t0
    sf.write(f'/workspace/tts_eval/fastpitch_{i+1}.wav', audio, 22050)
    print(f'  {elapsed:.3f}s -> fastpitch_{i+1}.wav')

# Rebuild listen.html with all 3 models
print('\n=== Rebuilding listen.html with FastPitch included ===')
import base64
wavdir = '/workspace/tts_eval'
files = sorted([f for f in os.listdir(wavdir) if f.endswith('.wav') and f != 'ref_audio.wav'])
html = '<html><body><h1>Norwegian TTS Comparison</h1>'
for f in files:
    with open(os.path.join(wavdir, f), 'rb') as fp2:
        b = base64.b64encode(fp2.read()).decode()
    html += f'<h3>{f}</h3><audio controls src="data:audio/wav;base64,{b}"></audio><br>'
html += '</body></html>'
with open('/workspace/tts_eval/listen.html', 'w') as fp2:
    fp2.write(html)
print(f'HTML: {len(html)} bytes with {len(files)} files')

# Push to GitHub
import shutil
shutil.copy('/workspace/tts_eval/listen.html', '/workspace/echotrail-nemo-tts/listen.html')
os.chdir('/workspace/echotrail-nemo-tts')
os.system('git add listen.html && git commit -m "Add FastPitch audio to eval" && git push')
print('\n=== ALL DONE ===')
PYEOF
