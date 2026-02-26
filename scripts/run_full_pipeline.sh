#!/bin/bash
set -e
WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
export DATA_DIR="$WORKSPACE/data"
export MODEL_DIR="$WORKSPACE/models"
MANIFEST_DIR="$DATA_DIR/manifests"

echo "EchoTrail NeMo TTS - Norwegian Voice Training"
echo "Workspace: $WORKSPACE"

pip install -q nemo_toolkit[tts] soundfile librosa tqdm matplotlib
mkdir -p "$DATA_DIR" "$MODEL_DIR" "$MANIFEST_DIR"

echo "=== Step 1: Download data ==="
bash "$WORKSPACE/scripts/01_download_data.sh"

echo ""
echo "=== Step 2: Prepare manifests ==="
python "$WORKSPACE/scripts/02_prepare_manifest.py"

echo ""
echo "=== Step 3: Extract supplementary data ==="
python "$WORKSPACE/scripts/03_extract_sup_data.py"

echo ""
echo "=== Step 4: Fine-tune FastPitch ==="
python "$WORKSPACE/scripts/04_train_fastpitch.py"

echo ""
echo "=== Step 5: Fine-tune HiFi-GAN ==="
python "$WORKSPACE/scripts/05_train_hifigan.py"

echo ""
echo "=== Step 6: Test inference ==="
python "$WORKSPACE/scripts/06_test_inference.py"

echo ""
echo "DONE - Models saved to $MODEL_DIR"
