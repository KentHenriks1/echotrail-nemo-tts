#!/usr/bin/env python3
"""
Step 3: Pre-process Norwegian text to IPA phonemes using espeak-ng.
Writes new manifest files with 'text' field replaced by IPA.
This avoids espeak-ng thread-safety crashes during NeMo training.
Uses subprocess calls (not library) for stability.
"""
import os, json, subprocess, sys
from pathlib import Path

# Ensure espeak library is findable
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/usr/lib/x86_64-linux-gnu/libespeak-ng.so"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(SCRIPT_DIR.parent / "data"))).resolve()
MANIFEST_DIR = DATA_DIR / "manifests"

BATCH_SIZE = 500  # Process texts in batches for efficiency


def text_to_ipa_batch(texts, language='nb'):
    """Convert a batch of texts to IPA using espeak-ng CLI (stable, no library crashes)."""
    from phonemizer import phonemize
    results = phonemize(
        texts,
        language=language,
        backend='espeak',
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
        njobs=1,  # Single job to avoid thread issues
    )
    if isinstance(results, str):
        results = [results]
    return results


def process_manifest(input_path, output_path):
    """Read manifest, convert text to IPA, write new manifest."""
    print(f"  Reading: {input_path}")
    with open(input_path) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    print(f"  {len(entries)} entries")

    # Process in batches
    texts = [e.get("text", "") for e in entries]
    all_ipa = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        ipa_batch = text_to_ipa_batch(batch)
        all_ipa.extend(ipa_batch)
        done = min(i + BATCH_SIZE, len(texts))
        print(f"  IPA converted: {done}/{len(texts)}", end='\r')
    print()

    # Verify
    assert len(all_ipa) == len(entries), f"Mismatch: {len(all_ipa)} IPA vs {len(entries)} entries"

    # Write new manifest with IPA text
    skipped = 0
    with open(output_path, 'w') as f:
        for entry, ipa in zip(entries, all_ipa):
            ipa = ipa.strip()
            if not ipa:
                skipped += 1
                continue
            entry["text"] = ipa
            entry["original_text"] = texts[entries.index(entry)] if entries.index(entry) < len(texts) else ""
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    total = len(entries) - skipped
    print(f"  Written: {output_path} ({total} entries, {skipped} skipped)")
    return total


def main():
    print("Step 3: Pre-process text to IPA")

    train_in = MANIFEST_DIR / "norwegian_train.json"
    val_in = MANIFEST_DIR / "norwegian_val.json"
    train_out = MANIFEST_DIR / "norwegian_train_ipa.json"
    val_out = MANIFEST_DIR / "norwegian_val_ipa.json"

    if not train_in.exists():
        print("Train manifest not found. Run steps 1-2 first.")
        return

    print("Processing training manifest")
    n_train = process_manifest(train_in, train_out)

    if val_in.exists():
        print("Processing validation manifest")
        n_val = process_manifest(val_in, val_out)
    else:
        print("No validation manifest found, skipping")
        n_val = 0

    print(f"\nDone! Train: {n_train}, Val: {n_val}")
    print("IPA manifests ready for training:")
    print(f"  {train_out}")
    print(f"  {val_out}")


if __name__ == "__main__":
    main()
