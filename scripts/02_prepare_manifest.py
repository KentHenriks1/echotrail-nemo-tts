#!/usr/bin/env python3
"""
Step 2: Parse NST HuggingFace JSON metadata + extracted MP3 audio into NeMo manifests.

NST HuggingFace naming: {pid}_{file_stem}.mp3
  Example: no99x069-22071999-1425_u0070005.mp3
  JSON has: pid="no99x069-22071999-1425", file="u0070005.wav"

Uses mutagen for fast MP3 duration reading (header-only, ~1ms per file).
"""
import os, json, random, re, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(SCRIPT_DIR.parent / "data")))
MANIFEST_DIR = DATA_DIR / "manifests"
TARGET_SR = 22050

def clean_text(text):
    """Clean NST transcription text: remove markup like \\Komma \\Punktum etc."""
    text = re.sub(r'\\[A-Za-zæøåÆØÅ]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_mp3_duration(path):
    """Get MP3 duration using mutagen (fast, header-only)."""
    try:
        from mutagen.mp3 import MP3
        return MP3(path).info.length
    except Exception:
        return None

def get_audio_duration(path):
    """Get audio duration - MP3 via mutagen, WAV via soundfile."""
    if path.endswith('.mp3'):
        return get_mp3_duration(path)
    try:
        import soundfile as sf
        return sf.info(path).duration
    except Exception:
        return None

def load_nst_metadata(data_dir):
    """Load NST JSON metadata and match with extracted MP3 files."""
    entries = []
    nst_dir = data_dir / "nst"
    if not nst_dir.exists():
        print(f"  {nst_dir} not found")
        return entries

    # Index all audio files by stem
    t0 = time.time()
    audio_files = {}
    for f in nst_dir.rglob("*.mp3"):
        audio_files[f.stem] = str(f)
    for f in nst_dir.rglob("*.wav"):
        audio_files[f.stem] = str(f)
    print(f"  Found {len(audio_files)} audio files ({time.time()-t0:.1f}s)")

    # Load JSON metadata shards
    json_files = sorted(nst_dir.glob("nst_no_train_close-*.json"))
    if not json_files:
        jsonl = nst_dir / "nst_tts_dataset.jsonl"
        if jsonl.exists():
            json_files = [jsonl]
    print(f"  Found {len(json_files)} metadata files")

    matched = 0
    skipped_no_audio = 0
    skipped_text = 0
    skipped_duration = 0
    t1 = time.time()

    for jf in json_files:
        print(f"  Processing {jf.name}...")
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
                    skipped_text += 1
                    continue

                # Build expected filename: {pid}_{file_stem}
                pid = rec.get("pid", "")
                raw_file = rec.get("file", "")
                file_stem = Path(raw_file).stem if raw_file else ""

                audio_path = None
                if pid and file_stem:
                    lookup = f"{pid}_{file_stem}"
                    if lookup in audio_files:
                        audio_path = audio_files[lookup]
                if not audio_path and file_stem:
                    if file_stem in audio_files:
                        audio_path = audio_files[file_stem]
                if not audio_path:
                    ch1 = rec.get("channel_1", "")
                    if ch1:
                        ch1_stem = Path(ch1).stem
                        if ch1_stem in audio_files:
                            audio_path = audio_files[ch1_stem]

                if not audio_path:
                    skipped_no_audio += 1
                    continue

                duration = get_audio_duration(audio_path)
                if duration is None or duration < 0.5 or duration > 30.0:
                    skipped_duration += 1
                    continue

                entries.append({
                    "audio_filepath": audio_path,
                    "text": text,
                    "duration": round(duration, 3),
                    "original_sr": 16000,
                    "speaker_id": rec.get("Speaker_ID", "unknown"),
                    "region": rec.get("Region_of_Youth", "unknown"),
                    "sex": rec.get("Sex", "unknown"),
                })
                matched += 1

                if matched % 10000 == 0:
                    print(f"    {matched} matched so far...")

    elapsed = time.time() - t1
    print(f"  Matched: {matched}, no_audio: {skipped_no_audio}, bad_text: {skipped_text}, bad_dur: {skipped_duration} ({elapsed:.1f}s)")
    return entries

def convert_to_wav(entries):
    """Convert MP3 to WAV at 22050Hz for NeMo."""
    import librosa, soundfile as sf
    wav_dir = DATA_DIR / "nst" / "wavs_22k"
    wav_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    failed = 0
    skipped = 0
    t0 = time.time()
    for i, entry in enumerate(entries):
        src = entry["audio_filepath"]
        stem = Path(src).stem
        dst = wav_dir / f"{stem}.wav"
        if dst.exists():
            entry["audio_filepath"] = str(dst)
            entry["original_sr"] = TARGET_SR
            skipped += 1
        else:
            try:
                audio, sr = librosa.load(src, sr=TARGET_SR)
                sf.write(str(dst), audio, TARGET_SR)
                entry["audio_filepath"] = str(dst)
                entry["original_sr"] = TARGET_SR
                converted += 1
            except Exception:
                failed += 1
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(entries) - i - 1) / rate
            print(f"    Progress: {i+1}/{len(entries)} ({rate:.0f}/s, ETA {eta:.0f}s)")
    elapsed = time.time() - t0
    print(f"  Done: {converted} converted, {skipped} cached, {failed} failed ({elapsed:.1f}s)")

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

    # Filter out entries where WAV doesn't exist
    entries = [e for e in entries if Path(e["audio_filepath"]).exists()]

    # Shuffle and split
    random.seed(42)
    random.shuffle(entries)
    split = int(len(entries) * 0.95)

    print("\nWriting manifests...")
    write_manifest(entries, "norwegian_all")
    write_manifest(entries[:split], "norwegian_train")
    write_manifest(entries[split:], "norwegian_val")

    # Single-speaker subset
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
