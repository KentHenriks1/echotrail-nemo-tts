#!/usr/bin/env python3
import os, json, random
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(SCRIPT_DIR.parent / "data")))
MANIFEST_DIR = DATA_DIR / "manifests"
TARGET_SR = 22050

def find_transcript_pairs(base_dir):
    import soundfile as sf
    pairs = []
    base = Path(base_dir)
    for wav_path in sorted(base.rglob("*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        lab_path = wav_path.with_suffix(".lab")
        text = None
        if txt_path.exists():
            text = txt_path.read_text(encoding="utf-8").strip()
        elif lab_path.exists():
            text = lab_path.read_text(encoding="utf-8").strip()
        if text and len(text) >= 3:
            try:
                info = sf.info(str(wav_path))
                if 0.5 < info.duration < 30.0:
                    pairs.append({"audio_filepath": str(wav_path), "text": text, "duration": round(info.duration, 3), "original_sr": info.samplerate})
            except Exception: pass
    for tsv in base.rglob("*.tsv"):
        try:
            with open(tsv, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        fn, text = parts[0], parts[1]
                        wp = base / fn
                        if not wp.suffix: wp = wp.with_suffix(".wav")
                        if wp.exists() and len(text) >= 3:
                            try:
                                info = sf.info(str(wp))
                                if 0.5 < info.duration < 30.0:
                                    pairs.append({"audio_filepath": str(wp), "text": text.strip(), "duration": round(info.duration, 3), "original_sr": info.samplerate})
                            except Exception: pass
        except Exception: pass
    return pairs

def resample_if_needed(pairs):
    import librosa, soundfile as sf
    resampled = 0
    for entry in pairs:
        if entry["original_sr"] != TARGET_SR:
            audio, sr = librosa.load(entry["audio_filepath"], sr=TARGET_SR)
            sf.write(entry["audio_filepath"], audio, TARGET_SR)
            resampled += 1
    if resampled > 0: print(f"  Resampled {resampled} files to {TARGET_SR}Hz")

def write_manifest(entries, name):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps({"audio_filepath": e["audio_filepath"], "text": e["text"], "duration": e["duration"]}, ensure_ascii=False) + "\n")
    total_dur = sum(e["duration"] for e in entries)
    print(f"  {path.name}: {len(entries)} utterances, {total_dur/60:.1f} min")

def main():
    print("Step 2: Prepare NeMo Manifests")
    all_pairs = []
    for subdir in ["nst_synth", "nb_tale"]:
        dp = DATA_DIR / subdir
        if dp.exists():
            print(f"Scanning {subdir}/")
            pairs = find_transcript_pairs(dp)
            print(f"  Found {len(pairs)} pairs")
            all_pairs.extend(pairs)
        else:
            print(f"{subdir}/ not found, skipping")
    if not all_pairs:
        print("No audio-transcript pairs found!")
        return
    print(f"Resampling to {TARGET_SR}Hz")
    resample_if_needed(all_pairs)
    random.seed(42)
    random.shuffle(all_pairs)
    split = int(len(all_pairs) * 0.95)
    print("Writing manifests")
    write_manifest(all_pairs, "norwegian_all")
    write_manifest(all_pairs[:split], "norwegian_train")
    write_manifest(all_pairs[split:], "norwegian_val")
    print(f"Manifests ready in {MANIFEST_DIR}/")

if __name__ == "__main__": main()
