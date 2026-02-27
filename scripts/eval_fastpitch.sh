#!/bin/bash
set -e
echo "=== FastPitch Norwegian Inference ==="
python3 << 'PYEOF'
import torch, os, soundfile as sf, time, glob, tempfile, tarfile, json
import numpy as np

os.makedirs('/workspace/tts_eval', exist_ok=True)
nemo_path = '/workspace/echotrail-nemo-tts/models/norwegian_fastpitch.nemo'

# Step 1: Extract .nemo to get config, weights, and tokenizer info
tmpdir = tempfile.mkdtemp()
with tarfile.open(nemo_path, 'r:') as tar:
    tar.extractall(tmpdir)

# List extracted contents
for root, dirs, files in os.walk(tmpdir):
    for f in files:
        print(f"  {os.path.join(root, f)}")

# Load weights
ckpt_files = glob.glob(f'{tmpdir}/**/model_weights.ckpt', recursive=True)
if not ckpt_files:
    ckpt_files = glob.glob(f'{tmpdir}/**/*.ckpt', recursive=True)
saved_state = torch.load(ckpt_files[0], map_location='cpu')

emb_key = 'fastpitch.encoder.word_emb.weight'
n_tokens = saved_state[emb_key].shape[0]
emb_dim = saved_state[emb_key].shape[1]
print(f'\nEmbedding: {n_tokens} tokens x {emb_dim} dim')

# Step 2: Build the SAME tokenizer used during training
# Read the IPA manifest to extract the character set
manifest_path = '/workspace/echotrail-nemo-tts/data/manifests/norwegian_train_ipa.json'
chars = set()
with open(manifest_path) as f:
    for line in f:
        entry = json.loads(line)
        for ch in entry.get('text', ''):
            chars.add(ch)

# Sort and create char-to-id mapping (same as training)
pad = '<pad>'
vocab = [pad] + sorted(chars)
char2id = {ch: i for i, ch in enumerate(vocab)}
print(f'Vocab size: {len(vocab)} (matches embedding: {len(vocab) == n_tokens})')
print(f'Vocab: {vocab[:20]}...')

# Step 3: Load pretrained English FastPitch and replace embedding + load weights
from nemo.collections.tts.models import FastPitchModel, HifiGanModel

fp = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
fp.fastpitch.encoder.word_emb = torch.nn.Embedding(n_tokens, emb_dim, padding_idx=0)
missing, unexpected = fp.load_state_dict(saved_state, strict=False)
print(f'Loaded weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}')
fp.eval().cuda()

# Step 4: Load HiFi-GAN vocoder (try multiple names)
hg = None
for name in ["tts_en_hifigan", "tts_en_lj_hifigan_ft_mixerttsx", "tts_hifigan"]:
    try:
        hg = HifiGanModel.from_pretrained(model_name=name)
        print(f'HiFi-GAN loaded: {name}')
        break
    except:
        continue

if hg is None:
    print('Downloading HiFi-GAN from NGC...')
    os.system('wget -q https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_lj_hifigan/versions/1.6.0/files/tts_en_lj_hifigan.nemo -O /tmp/hifigan.nemo')
    hg = HifiGanModel.restore_from('/tmp/hifigan.nemo')

hg.eval().cuda()

# Step 5: IPA conversion + manual tokenization + inference
from phonemizer import phonemize

def text_to_tokens(text, char2id):
    """Convert IPA text to token IDs using our custom mapping."""
    ids = []
    for ch in text:
        if ch in char2id:
            ids.append(char2id[ch])
        # Skip unknown characters silently
    return ids

texts = [
    'Hei, jeg heter EchoTrail og jeg kan hjelpe deg med aa finne veien.',
    'Det er viktig aa ta vare paa naturen rundt oss.',
    'Godmorgen! Hvordan har du det i dag?',
    'Norge er et vakkert land med fjorder, fjell og nordlys.',
    'Denne setningen tester om modellen haandterer lengre tekst med flere ord og naturlig prosodi.',
]

for i, text in enumerate(texts):
    # Convert to IPA
    ipa = phonemize(text, language='nb', backend='espeak', strip=True,
                    preserve_punctuation=True, language_switch='remove-flags')
    print(f'\n[{i+1}/5] Text: {text}')
    print(f'  IPA: {ipa}')

    # Tokenize with OUR tokenizer
    token_ids = text_to_tokens(ipa, char2id)
    print(f'  Tokens: {len(token_ids)} ids, max={max(token_ids) if token_ids else 0}')

    t0 = time.time()
    with torch.no_grad():
        tokens_tensor = torch.tensor([token_ids], dtype=torch.long).cuda()
        # Call FastPitch directly with token tensor
        spec = fp.fastpitch.infer(text=tokens_tensor, text_len=torch.tensor([len(token_ids)]).cuda())
        # spec is a tuple, first element is the mel spectrogram
        if isinstance(spec, tuple):
            mel = spec[0]
        else:
            mel = spec
        audio = hg.convert_spectrogram_to_audio(spec=mel).squeeze().cpu().numpy()

    elapsed = time.time() - t0
    out_path = f'/workspace/tts_eval/fastpitch_{i+1}.wav'
    sf.write(out_path, audio, 22050)
    print(f'  {elapsed:.3f}s -> fastpitch_{i+1}.wav')

print('\n=== FastPitch DONE ===')
os.system("ls -la /workspace/tts_eval/fastpitch_*.wav")
PYEOF
