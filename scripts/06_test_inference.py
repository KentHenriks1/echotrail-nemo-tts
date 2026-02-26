#!/usr/bin/env python3
import os, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(SCRIPT_DIR.parent / "models")))
OUTPUT_DIR = MODEL_DIR / "test_samples"

TEST_SENTENCES = [
    "Velkommen til fjellet. Her oppe kan du se hele dalen.",
    "Vi starter turen fra Jotunheimen og gar mot Besseggen.",
    "Toppen ligger pa tusen fire hundre og femti meter over havet.",
    "Fjellbjorka klamrer seg til steinene, og vinden baerer med seg lukten av lyng.",
    "Stien snor seg oppover gjennom den tette bjorkeskogen. Sollyset filtreres gjennom bladverket.",
    "Plutselig apner landskapet seg, og der, langt der nede, ligger fjorden som et blatt speil.",
    "Stopp. Lytt. Horer du det? Stillheten har sin egen melodi her oppe.",
    "Gammelstien folger elva nedover mot sjoen der naust og brygger vitner om svunne tider.",
]

def main():
    print("Step 6: Test Norwegian TTS Inference")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fp_path = MODEL_DIR / "norwegian_fastpitch.nemo"
    hg_path = MODEL_DIR / "norwegian_hifigan.nemo"
    try:
        import torch, soundfile as sf
        from nemo.collections.tts.models import FastPitchModel, HifiGanModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Loading models")
        fp = (FastPitchModel.restore_from(str(fp_path)) if fp_path.exists() else FastPitchModel.from_pretrained("tts_en_fastpitch")).to(device).eval()
        hg = (HifiGanModel.restore_from(str(hg_path)) if hg_path.exists() else HifiGanModel.from_pretrained("tts_en_hifigan")).to(device).eval()
        print(f"Generating {len(TEST_SENTENCES)} test samples")
        for i, text in enumerate(TEST_SENTENCES):
            try:
                start = time.time()
                with torch.no_grad():
                    parsed = fp.parse(text)
                    spec = fp.generate_spectrogram(tokens=parsed)
                    audio = hg.convert_spectrogram_to_audio(spec=spec)
                audio_np = audio.squeeze().cpu().numpy()
                lat = int((time.time() - start) * 1000)
                wav_path = OUTPUT_DIR / f"test_{i:02d}.wav"
                sf.write(str(wav_path), audio_np, 22050)
                print(f"  [{i+1}] {lat}ms | {len(audio_np)/22050:.1f}s | {wav_path.name} | {text[:50]}")
            except Exception as e:
                print(f"  [{i+1}] FAILED: {e}")
        print(f"Samples saved to {OUTPUT_DIR}/")
    except ImportError as e:
        print(f"NeMo not available: {e}")

if __name__ == "__main__": main()
