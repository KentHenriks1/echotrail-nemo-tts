#!/usr/bin/env python3
"""
Step 2: Parse NST HuggingFace JSON metadata + extracted MP3 audio into NeMo manifests.
NST format: JSONL with {text, Speaker_ID, Region_of_Youth, Sex, file, t0, t1, t2, ...}
Audio: MP3 files extracted from tar.gz shards into data/nst/
"""
import os, json, random, re, glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(SCRIPT_DIR.parent / "data")))
MANIFEST_DIR = DATA_DIR / "manifests"
TARGET_SR = 22050

def clean_text(text):
    """Clean NST transcription text: remove markup like \\Komma \\Punktum etc."""
    text = re.sub(r'\\[A-Za-zæøåÆØÅ]+', '', text)  # Remove \Komma \Punktum etc
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_nst_metadata(data_dir):
    """Load all NST JSON metadata shards and match with extracted audio files."""
    import soundfile as sf
    entries = []
    nst_dir = data_dir / "nst"
    if not nst_dir.exists():
        print(f"  {nst_dir} not found")
        return entries

    # Find all audio files (MP3 from tar.gz extraction)
    audio_files = {}
    for ext in ["*.mp3", "*.wav"]:
        for f in nst_dir.rglob(ext):
            audio_files[f.name] = str(f)
            # Also index without extension
            audio_files[f.stem] = str(f)
    print(f"  Found {len(audio_files)//2} audio files")

    # Load JSON metadata shards
    json_files = sorted(nst_dir.glob("nst_no_train_close-*.json"))
    if not json_files:
        # Try JSONL from Sprakbanken
        jsonl = nst_dir / "nst_tts_dataset.jsonl"
        if jsonl.exists():
            json_files = [jsonl]
    print(f"  Found {len(json_files)} metadata files")

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = clean_text(rec.get("text", ""))
                if not text or len(text) < 5:
                    continue

                # Find audio file - try multiple name patterns
                audio_path = None
                for key in [rec.get("file", ""), rec.get("channel_1", "")]:
                    if not key:
                        continue
                    basename = Path(key).name
                    stem = Path(key).stem
                    # Try exact name, then stem with .mp3
                    for lookup in [basename, stem, f"{stem}.mp3", f"{stem}.wav"]:
                        if lookup in audio_files:
                            audio_path = audio_files[lookup]
                            break
                    if audio_path:
                        break

                if not audio_path:
                    continue

                # Get duration from audio file
                try:
                    info = sf.info(audio_path)
                    duration = info.duration
                    if duration < 0.5 or duration > 30.0:
                        continue
                except Exception:
                    continue

                entries.append({
                    "audio_filepath": audio_path,
                    "text": text,
                    "duration": round(duration, 3),
                    "original_sr": info.samplerate,
                    "speaker_id": rec.get("Speaker_ID", "unknown"),
                    "region": rec.get("Region_of_Youth", "unknown"),
                    "sex": rec.get("Sex", "unknown"),
                })

    return entries

def convert_to_wav(entries):
    """Convert MP3 to WAV at 22050Hz for NeMo."""
    import librosa, soundfile as sf
    wav_dir = DATA_DIR / "nst" / "wavs_22k"
    wav_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    for entry in entries:
        src = entry["audio_filepath"]
        stem = Path(src).stem
        dst = wav_dir / f"{stem}.wav"
        if not dst.exists():
            try:
                audio, sr = librosa.load(src, sr=TARGET_SR)
                sf.write(str(dst), audio, TARGET_SR)
                converted += 1
            except Exception:
                continue
        entry["audio_filepath"] = str(dst)
        entry["original_sr"] = TARGET_SR
    print(f"  Converted {converted} files to {TARGET_SR}Hz WAV")

def write_manifest(entries, name):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps({
                "audio_filepath": e["audio_filepath"],
                "text": e["text"],
                "duration": e["duration"]
            }, ensure_ascii=False) + "\n")
    total_dur = sum(e["duration"] for e in entries)
    print(f"  {path.name}: {len(entries)} utterances, {total_dur/60:.1f} min")

def main():
    print("Step 2: Prepare NeMo Manifests from NST data")
    entries = load_nst_metadata(DATA_DIR)

    if not entries:
        print("No audio-transcript pairs found!")
        print("  Make sure step 1 downloaded and extracted data to data/nst/")
        return

    total_dur = sum(e["duration"] for e in entries)
    unique_speakers = len(set(e["speaker_id"] for e in entries))
    regions = set(e["region"] for e in entries)
    print(f"\n  Total: {len(entries)} utterances, {total_dur/60:.1f} min")
    print(f"  Speakers: {unique_speakers}")
    print(f"  Regions: {', '.join(sorted(regions)[:10])}")

    # Convert MP3 to WAV 22050Hz
    print("\nConverting to 22050Hz WAV...")
    convert_to_wav(entries)

    # Shuffle and split
    random.seed(42)
    random.shuffle(entries)
    split = int(len(entries) * 0.95)

    print("\nWriting manifests...")
    write_manifest(entries, "norwegian_all")
    write_manifest(entries[:split], "norwegian_train")
    write_manifest(entries[split:], "norwegian_val")

    # Also create single-speaker subset (best speaker)
    speaker_counts = {}
    for e in entries:
        sid = e["speaker_id"]
        speaker_counts[sid] = speaker_counts.get(sid, 0) + 1
    best_speaker = max(speaker_counts, key=speaker_counts.get)
    ss_entries = [e for e in entries if e["speaker_id"] == best_speaker]
    if len(ss_entries) >= 50:
        write_manifest(ss_entries, "norwegian_single_speaker")
        ss_dur = sum(e["duration"] for e in ss_entries)
        print(f"  Best speaker: {best_speaker} ({len(ss_entries)} utt, {ss_dur/60:.1f} min)")

    print(f"\nManifests ready in {MANIFEST_DIR}/")

if __name__ == "__main__":
    main()
