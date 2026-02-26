#!/usr/bin/env python3
import os, json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(SCRIPT_DIR.parent / "data")))
MANIFEST_DIR = DATA_DIR / "manifests"
SUP_DATA_DIR = DATA_DIR / "sup_data"

def main():
    print("Step 3: Extract Supplementary Data")
    train_manifest = MANIFEST_DIR / "norwegian_train.json"
    val_manifest = MANIFEST_DIR / "norwegian_val.json"
    if not train_manifest.exists():
        print("Train manifest not found. Run step 2 first.")
        return
    SUP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(train_manifest) as f:
        train_count = sum(1 for _ in f)
    print(f"  Train manifest: {train_count} utterances")
    try:
        import torch, librosa, numpy as np, soundfile as sf
        from nemo.collections.tts.models import FastPitchModel
        print("Downloading pretrained FastPitch (English baseline)")
        pretrained = FastPitchModel.from_pretrained("tts_en_fastpitch")
        print("  Pretrained model loaded")
        print("Computing pitch statistics")
        pitches = []
        for mp in [train_manifest, val_manifest]:
            if not mp.exists(): continue
            with open(mp) as f:
                for line in f:
                    entry = json.loads(line)
                    try:
                        audio, sr = librosa.load(entry["audio_filepath"], sr=22050)
                        f0, _, _ = librosa.pyin(audio, fmin=50, fmax=600, sr=sr, frame_length=1024, hop_length=256)
                        voiced = f0[~np.isnan(f0)]
                        if len(voiced) > 0: pitches.extend(voiced.tolist())
                    except Exception: pass
        if pitches:
            pm, ps = np.mean(pitches), np.std(pitches)
            print(f"  Pitch mean: {pm:.1f} Hz, std: {ps:.1f} Hz")
            with open(SUP_DATA_DIR / "pitch_stats.json", "w") as f:
                json.dump({"pitch_mean": float(pm), "pitch_std": float(ps)}, f, indent=2)
        print(f"Supplementary data ready in {SUP_DATA_DIR}/")
    except ImportError as e:
        print(f"NeMo not available ({e}), skipping - will extract during training")

if __name__ == "__main__": main()
