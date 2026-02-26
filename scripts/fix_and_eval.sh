#!/bin/bash
# Fix torchvision conflict and run eval step by step
set -e

echo "=== Step 1: Fix torchvision ==="
# Get PyTorch version and reinstall matching torchvision
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
echo "PyTorch version: $TORCH_VERSION"
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
echo "CUDA version: $CUDA_VERSION"

# Reinstall torchvision matching current PyTorch
pip install --force-reinstall torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3

echo ""
echo "=== Step 2: Test FastPitch inference ==="
python3 -c "
import torch, os, soundfile as sf
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
from phonemizer import phonemize

os.makedirs('/workspace/tts_eval', exist_ok=True)

fp = FastPitchModel.restore_from('/workspace/echotrail-nemo-tts/models/norwegian_fastpitch.nemo')
fp.eval().cuda()
hg = HifiGanModel.from_pretrained('nvidia/tts_en_hifigan')
hg.eval().cuda()

texts = [
    'Hei, jeg heter EchoTrail og jeg kan hjelpe deg med aa finne veien.',
    'Det er viktig aa ta vare paa naturen rundt oss.',
    'Godmorgen! Hvordan har du det i dag?',
    'Norge er et vakkert land med fjorder, fjell og nordlys.',
    'Denne setningen tester om modellen haandterer lengre tekst med flere ord og naturlig prosodi.',
]

import time
for i, text in enumerate(texts):
    ipa = phonemize(text, language='nb', backend='espeak', strip=True, preserve_punctuation=True, language_switch='remove-flags')
    print(f'Text: {text}')
    print(f'IPA:  {ipa}')
    t0 = time.time()
    with torch.no_grad():
        spec = fp.generate_spectrogram(tokens=fp.parse(ipa))
        audio = hg.convert_spectrogram_to_audio(spec=spec).squeeze().cpu().numpy()
    print(f'Time: {time.time()-t0:.3f}s')
    sf.write(f'/workspace/tts_eval/fastpitch_{i+1}.wav', audio, 22050)
    print(f'Saved: /workspace/tts_eval/fastpitch_{i+1}.wav')
    print()

del fp, hg
torch.cuda.empty_cache()
print('FastPitch DONE')
"

echo ""
echo "=== Step 3: Install and test F5-TTS ==="
pip install f5-tts 2>&1 | tail -3
python3 -c "
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
            print(f'Time: {time.time()-t0:.3f}s')
            sf.write(f'/workspace/tts_eval/f5tts_{i+1}.wav', wav, sr)
            print(f'Saved: /workspace/tts_eval/f5tts_{i+1}.wav')
            print()
    del model
    torch.cuda.empty_cache()
    print('F5-TTS DONE')
except Exception as e:
    print(f'F5-TTS ERROR: {e}')
"

echo ""
echo "=== Step 4: Install and test Chatterbox ==="
pip install chatterbox-tts 2>&1 | tail -3
python3 -c "
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
        print(f'Time: {time.time()-t0:.3f}s')
        ta.save(f'/workspace/tts_eval/chatterbox_{i+1}.wav', wav, model.sr)
        print(f'Saved: /workspace/tts_eval/chatterbox_{i+1}.wav')
        print()
    del model
    torch.cuda.empty_cache()
    print('Chatterbox DONE')
except Exception as e:
    print(f'Chatterbox ERROR: {e}')
"

echo ""
echo "=== RESULTS ==="
ls -la /workspace/tts_eval/*.wav 2>/dev/null
echo "Done! Listen to the WAV files to compare."
