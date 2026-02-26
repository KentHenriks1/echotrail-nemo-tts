#!/bin/bash
# Step 1: Download Norwegian speech data from HuggingFace (NbAiLab/NST)
# Close microphone, Norwegian, 9 shards x ~511MB = ~4.5GB total
# We download 2 shards (~1GB) for initial training
set -e

DATA_DIR="${DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data}"
mkdir -p "$DATA_DIR/nst"

echo "== Downloading NST Norwegian Speech (close mic) from HuggingFace =="
echo "   Source: NbAiLab/NST (Apache 2.0)"
echo "   Data dir: $DATA_DIR/nst"

# Download NST TTS metadata from Sprakbanken (single-speaker, 5363 utterances)
echo ""
echo "-- Downloading NST TTS metadata --"
wget -q --show-progress -O "$DATA_DIR/nst/nst_tts_dataset.jsonl" \
  "https://www.nb.no/sbfil/talesyntese/nst_tts_dataset.jsonl" 2>&1 || true
if [ -f "$DATA_DIR/nst/nst_tts_dataset.jsonl" ]; then
    echo "   OK: $(wc -l < "$DATA_DIR/nst/nst_tts_dataset.jsonl") entries"
fi

# Download close-mic shards from HuggingFace
SHARDS_TO_DOWNLOAD=2
BASE_URL="https://huggingface.co/datasets/NbAiLab/NST/resolve/main/data/train"

for i in $(seq -w 1 $SHARDS_TO_DOWNLOAD); do
    SHARD="nst_no_train_close-000${i}-of-0009"

    if [ ! -f "$DATA_DIR/nst/${SHARD}.tar.gz" ]; then
        echo ""
        echo "-- Downloading shard $i/$SHARDS_TO_DOWNLOAD: ${SHARD}.tar.gz (~511MB) --"
        wget -q --show-progress -O "$DATA_DIR/nst/${SHARD}.tar.gz" \
          "${BASE_URL}/${SHARD}.tar.gz?download=true" 2>&1
        echo "   OK: $(du -h "$DATA_DIR/nst/${SHARD}.tar.gz" | cut -f1)"
    else
        echo "   Shard $i already downloaded"
    fi

    if [ ! -f "$DATA_DIR/nst/${SHARD}.json" ]; then
        echo "   Downloading metadata: ${SHARD}.json"
        wget -q --show-progress -O "$DATA_DIR/nst/${SHARD}.json" \
          "${BASE_URL}/${SHARD}.json?download=true" 2>&1
    fi
done

# Extract audio shards
echo ""
echo "-- Extracting audio files --"
for i in $(seq -w 1 $SHARDS_TO_DOWNLOAD); do
    SHARD="nst_no_train_close-000${i}-of-0009"
    if [ -f "$DATA_DIR/nst/${SHARD}.tar.gz" ] && [ ! -f "$DATA_DIR/nst/.extracted_${i}" ]; then
        echo "   Extracting shard $i..."
        tar -xzf "$DATA_DIR/nst/${SHARD}.tar.gz" -C "$DATA_DIR/nst/" 2>/dev/null || true
        touch "$DATA_DIR/nst/.extracted_${i}"
        echo "   OK"
    fi
done

echo ""
echo "-- Data inventory --"
AUDIO_COUNT=$(find "$DATA_DIR/nst" -name '*.mp3' -o -name '*.wav' 2>/dev/null | wc -l)
JSON_COUNT=$(find "$DATA_DIR/nst" -name '*.json' 2>/dev/null | wc -l)
echo "   Audio files: $AUDIO_COUNT"
echo "   JSON metadata files: $JSON_COUNT"
echo "   Total size: $(du -sh "$DATA_DIR/nst" | cut -f1)"
