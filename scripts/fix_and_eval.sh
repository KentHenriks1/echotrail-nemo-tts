#!/bin/bash
set -e

echo "=== Step 1: Fix torchvision ==="
pip install --force-reinstall torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3

echo ""
echo "=== Step 2: Test FastPitch inference ==="
python3 << 'PYEOF'
import torch, os, soundfile as sf, time, glob, tempfile, tarfile

os.makedirs('/workspace/tts_eval', exist_ok=True)
nemo_path = '/workspace/echotrail-nemo-tts/models/norwegian_fastpitch.nemo'

# Extract .nemo archive to get weights
tmpdir = tempfile.mkdtemp()
with tarfile.open(nemo_path, 'r:') as tar:
    tar.extractall(tmpdir)

ckpt_files = glob.glob(f'{tmpdir}/**/model_weights.ckpt', recursive=True)
if not ckpt_files:
    ckpt_files = glob.glob(f'{tmpdir}/**/*.ckpt', recursive=True)
saved_state = torch.load(ckpt_files[0], map_location='cuda')

emb_key = 'fastpitch.encoder.word_emb.weight'
n_tokens = saved_state[emb_key].shape[0]
emb_dim = saved_state[emb_key].shape[1]
print(f'Saved embedding: {n_tokens} tokens x {emb_dim} dim')

from nemo.collections.tts.models import FastPitchModel, HifiGanModel

# Load pretrained English FastPitch, resize embedding, load our weights
fp = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch")
fp.fastpitch.encoder.word_emb = torch.nn.Embedding(n_tokens, emb_dim, padding_idx=0)
fp.load_state_dict(saved_state, strict=False)
fp.eval().cuda()
print(f'FastPitch loaded! Embedding: {fp.fastpitch.encoder.word_emb.weight.shape}')

# Load HiFi-GAN - try multiple names since some require auth
hg = None
for name in ["tts_en_hifigan", "tts_en_lj_hifigan_ft_mixerttsx", "tts_hifigan"]:
    try:
        print(f'Trying HiFi-GAN: {name}')
        hg = HifiGanModel.from_pretrained(model_name=name)
        print(f'HiFi-GAN loaded: {name}')
        break
    except Exception as e:
        print(f'  Failed: {e}')

if hg is None:
    # Last resort: download from NGC directly
    print('Downloading HiFi-GAN from NGC...')
    os.system('wget -q https://api.ngc.nvidia.com/v2/models/nvidia/nemo/tts_en_lj_hifigan/versions/1.6.0/files/tts_en_lj_hifigan.nemo -O /tmp/hifigan.nemo')
    hg = HifiGanModel.restore_from('/tmp/hifigan.nemo')
    print('HiFi-GAN loaded from NGC')

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
    ipa = phonemize(text, language='nb', backend='espeak', strip=True, preserve_punctuation=True, language_switch='remove-flags')
    print(f'Text: {text}')
    print(f'IPA:  {ipa}')
    t0 = time.time()
    with torch.no_grad():
        spec = fp.generate_spectrogram(tokens=fp.parse(ipa))
        audio = hg.convert_spectrogram_to_audio(spec=spec).squeeze().cpu().numpy()
    elapsed = time.time() - t0
    sf.write(f'/workspace/tts_eval/fastpitch_{i+1}.wav', audio, 22050)
    print(f'Saved: /workspace/tts_eval/fastpitch_{i+1}.wav ({elapsed:.3f}s)\n')

del fp, hg
torch.cuda.empty_cache()
print('FastPitch DONE')
PYEOF

echo ""
echo "=== Step 3: Install and test F5-TTS ==="
pip install f5-tts 2>&1 | tail -3
python3 << 'PYEOF'
import torch, os, soundfile as sf, glob, time
os.makedirs('/workspace/tts_eval', exist_ok=True)

try:
    from f5_tts.api import F5TTS
    model = F5TTS(model_type='F5-TTS', ckpt_file='hf://akhbar/F5_Norwegian', device='cuda')

    ref_wavs = sorted(glob.glob('/workspace/echotrail-nemo-tts/data/wavs/*.wav'))
    ref_audio = ref_wavs[0] if ref_wavs else None

    if ref_audio:
        texts = [
            'Hei, jeg heter EchoTrail og jeg kan hjelpe deg med aa finne veien.',
            'Det er viktig aa ta vare paa naturen rundt oss.',
            'Godmorgen! Hvordan har du det i dag?',
            'Norge er et vakkert land med fjorder, fjell og nordlys.',
            'Denne setningen tester om modellen haandterer lengre tekst med flere ord og naturlig prosodi.',
        ]
        for i, text in enumerate(texts):
            print(f'Text: {text}')
            t0 = time.time()
            wav, sr, _ = model.infer(ref_file=ref_audio, ref_text='referanse', gen_text=text)
            elapsed = time.time() - t0
            sf.write(f'/workspace/tts_eval/f5tts_{i+1}.wav', wav, sr)
            print(f'Saved: /workspace/tts_eval/f5tts_{i+1}.wav ({elapsed:.3f}s)\n')
    del model
    torch.cuda.empty_cache()
    print('F5-TTS DONE')
except Exception as e:
    print(f'F5-TTS ERROR: {e}')
    import traceback; traceback.print_exc()
PYEOF

echo ""
echo "=== Step 4: Install and test Chatterbox ==="
pip install chatterbox-tts 2>&1 | tail -3
python3 << 'PYEOF'
import torch, os, time
os.makedirs('/workspace/tts_eval', exist_ok=True)

try:
    from pathlib import Path
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS
    from huggingface_hub import hf_hub_download

    REPO_ID = 'akhbar/chatterbox-tts-norwegian'
    for f in ['ve.safetensors', 't3_cfg.safetensors', 's3gen.safetensors', 'tokenizer.json', 'conds.pt']:
        lp = hf_hub_download(repo_id=REPO_ID, filename=f)

    model = ChatterboxTTS.from_local(Path(lp).parent, device='cuda')

    texts = [
        'Hei, jeg heter EchoTrail og jeg kan hjelpe deg med aa finne veien.',
        'Det er viktig aa ta vare paa naturen rundt oss.',
        'Godmorgen! Hvordan har du det i dag?',
        'Norge er et vakkert land med fjorder, fjell og nordlys.',
        'Denne setningen tester om modellen haandterer lengre tekst med flere ord og naturlig prosodi.',
    ]
    for i, text in enumerate(texts):
        print(f'Text: {text}')
        t0 = time.time()
        wav = model.generate(text, exaggeration=1.0, cfg_weight=0.5, temperature=0.4)
        elapsed = time.time() - t0
        ta.save(f'/workspace/tts_eval/chatterbox_{i+1}.wav', wav, model.sr)
        print(f'Saved: /workspace/tts_eval/chatterbox_{i+1}.wav ({elapsed:.3f}s)\n')
    del model
    torch.cuda.empty_cache()
    print('Chatterbox DONE')
except Exception as e:
    print(f'Chatterbox ERROR: {e}')
    import traceback; traceback.print_exc()
PYEOF

echo ""
echo "=== RESULTS ==="
ls -la /workspace/tts_eval/*.wav 2>/dev/null
echo "Done! Listen to WAV files to compare."
