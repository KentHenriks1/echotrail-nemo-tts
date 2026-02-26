#!/usr/bin/env python3
"""
Side-by-side evaluation of 3 Norwegian TTS models:
1. Our FastPitch (NeMo) - trained on 40h NST data
2. akhbar/F5_Norwegian - F5-TTS trained on 6000h Sprakbanken
3. akhbar/chatterbox-tts-norwegian - Chatterbox fine-tuned on 6000h

Run on RunPod with GPU. Generates WAV files for comparison.
"""

import os
import time
import json

OUTPUT_DIR = "/workspace/tts_eval"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_SENTENCES = [
    "Hei, jeg heter EchoTrail og jeg kan hjelpe deg med aa finne veien.",
    "Det er viktig aa ta vare paa naturen rundt oss.",
    "Godmorgen! Hvordan har du det i dag?",
    "Norge er et vakkert land med fjorder, fjell og nordlys.",
    "Denne setningen tester om modellen haandterer lengre tekst med flere ord og naturlig prosodi.",
]

results = {}

# ============================================================
# MODEL 1: Our FastPitch + HiFi-GAN
# ============================================================
print("=" * 60)
print("MODEL 1: FastPitch (our trained model)")
print("=" * 60)

try:
    import torch
    import soundfile as sf
    from nemo.collections.tts.models import FastPitchModel, HifiGanModel

    fastpitch_path = "/workspace/echotrail-nemo-tts/models/norwegian_fastpitch.nemo"
    if os.path.exists(fastpitch_path):
        fp_model = FastPitchModel.restore_from(fastpitch_path)
        fp_model.eval()
        fp_model = fp_model.cuda()

        hifigan = HifiGanModel.from_pretrained("nvidia/tts_en_hifigan")
        hifigan.eval()
        hifigan = hifigan.cuda()

        from phonemizer import phonemize

        fastpitch_times = []
        for i, text in enumerate(TEST_SENTENCES):
            ipa_text = phonemize(
                text,
                language='nb',
                backend='espeak',
                strip=True,
                preserve_punctuation=True,
                language_switch='remove-flags',
            )
            print(f"  Text: {text}")
            print(f"  IPA:  {ipa_text}")

            t0 = time.time()
            with torch.no_grad():
                parsed = fp_model.parse(ipa_text)
                spectrogram = fp_model.generate_spectrogram(tokens=parsed)
                audio = hifigan.convert_spectrogram_to_audio(spec=spectrogram)
                audio_np = audio.squeeze().cpu().numpy()

            elapsed = time.time() - t0
            fastpitch_times.append(elapsed)

            out_path = os.path.join(OUTPUT_DIR, f"fastpitch_{i+1}.wav")
            sf.write(out_path, audio_np, 22050)
            print(f"  Saved: {out_path} ({elapsed:.2f}s)")

        results["fastpitch"] = {
            "avg_time": sum(fastpitch_times) / len(fastpitch_times),
            "total_time": sum(fastpitch_times),
            "status": "success",
        }
        print(f"\n  FastPitch avg inference: {results['fastpitch']['avg_time']:.3f}s\n")

        del fp_model, hifigan
        torch.cuda.empty_cache()
    else:
        print(f"  ERROR: Model not found at {fastpitch_path}")
        results["fastpitch"] = {"status": "model_not_found"}

except Exception as e:
    print(f"  ERROR: {e}")
    results["fastpitch"] = {"status": f"error: {e}"}


# ============================================================
# MODEL 2: F5-TTS Norwegian
# ============================================================
print("=" * 60)
print("MODEL 2: F5-TTS Norwegian (akhbar/F5_Norwegian)")
print("=" * 60)

try:
    import torch
    torch.cuda.empty_cache()

    from f5_tts.api import F5TTS

    f5_model = F5TTS(
        model_type="F5-TTS",
        ckpt_file="hf://akhbar/F5_Norwegian",
        device="cuda",
    )

    import glob
    ref_wavs = sorted(glob.glob("/workspace/echotrail-nemo-tts/data/wavs/*.wav"))
    if ref_wavs:
        ref_audio = ref_wavs[0]
        ref_text = "referanse audio"
    else:
        ref_audio = None

    if ref_audio:
        f5_times = []
        for i, text in enumerate(TEST_SENTENCES):
            print(f"  Text: {text}")
            t0 = time.time()

            wav_path = os.path.join(OUTPUT_DIR, f"f5tts_{i+1}.wav")
            wav, sr, _ = f5_model.infer(
                ref_file=ref_audio,
                ref_text=ref_text,
                gen_text=text,
            )

            elapsed = time.time() - t0
            f5_times.append(elapsed)

            import soundfile as sf
            sf.write(wav_path, wav, sr)
            print(f"  Saved: {wav_path} ({elapsed:.2f}s)")

        results["f5_norwegian"] = {
            "avg_time": sum(f5_times) / len(f5_times),
            "total_time": sum(f5_times),
            "status": "success",
        }
        print(f"\n  F5-TTS avg inference: {results['f5_norwegian']['avg_time']:.3f}s\n")

        del f5_model
        torch.cuda.empty_cache()
    else:
        print("  ERROR: No reference audio found")
        results["f5_norwegian"] = {"status": "no_ref_audio"}

except Exception as e:
    print(f"  ERROR: {e}")
    results["f5_norwegian"] = {"status": f"error: {e}"}


# ============================================================
# MODEL 3: Chatterbox Norwegian
# ============================================================
print("=" * 60)
print("MODEL 3: Chatterbox Norwegian (akhbar/chatterbox-tts-norwegian)")
print("=" * 60)

try:
    import torch
    torch.cuda.empty_cache()

    from pathlib import Path
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS
    from huggingface_hub import hf_hub_download

    REPO_ID = "akhbar/chatterbox-tts-norwegian"
    for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
        local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

    model = ChatterboxTTS.from_local(Path(local_path).parent, device="cuda")

    cb_times = []
    for i, text in enumerate(TEST_SENTENCES):
        print(f"  Text: {text}")
        t0 = time.time()

        wav = model.generate(text, exaggeration=1.0, cfg_weight=0.5, temperature=0.4)

        elapsed = time.time() - t0
        cb_times.append(elapsed)

        out_path = os.path.join(OUTPUT_DIR, f"chatterbox_{i+1}.wav")
        ta.save(out_path, wav, model.sr)
        print(f"  Saved: {out_path} ({elapsed:.2f}s)")

    results["chatterbox_norwegian"] = {
        "avg_time": sum(cb_times) / len(cb_times),
        "total_time": sum(cb_times),
        "status": "success",
    }
    print(f"\n  Chatterbox avg inference: {results['chatterbox_norwegian']['avg_time']:.3f}s\n")

    del model
    torch.cuda.empty_cache()

except Exception as e:
    print(f"  ERROR: {e}")
    results["chatterbox_norwegian"] = {"status": f"error: {e}"}


# ============================================================
# SUMMARY
# ============================================================
print("=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)

for name, res in results.items():
    status = res.get("status", "unknown")
    if status == "success":
        print(f"  {name:30s} | avg: {res['avg_time']:.3f}s | status: OK")
    else:
        print(f"  {name:30s} | status: {status}")

print(f"\nWAV files saved to: {OUTPUT_DIR}/")
print("Listen and compare:")
print(f"  ls -la {OUTPUT_DIR}/")

with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results: {OUTPUT_DIR}/results.json")
