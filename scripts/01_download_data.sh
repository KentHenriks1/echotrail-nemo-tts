#!/bin/bash
set -e
DATA_DIR="${DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data}"
mkdir -p "$DATA_DIR/nst_synth" "$DATA_DIR/nb_tale"

echo "-- Downloading NST Norwegian Speech Synthesis (sbr-15) --"
echo "   Single male speaker, 5363 utterances, CC-0"

NST_URL="https://www.nb.no/sbfil/talesyntese/16kHz_2020/no_16khz.tar.gz"
NST_TAR="$DATA_DIR/nst_synth/no_16khz.tar.gz"

if [ ! -f "$DATA_DIR/nst_synth/.done" ]; then
    echo "   Downloading NST Speech Synthesis..."
    wget -q --show-progress -O "$NST_TAR" "$NST_URL" 2>&1 || {
        NST_URL_ALT="https://www.nb.no/sbfil/talesyntese/no_16khz.tar.gz"
        wget -q --show-progress -O "$NST_TAR" "$NST_URL_ALT" 2>&1 || {
            echo "   FAILED: Manual download from https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-15/"
        }
    }
    if [ -f "$NST_TAR" ] && [ -s "$NST_TAR" ]; then
        echo "   Extracting..."
        tar -xzf "$NST_TAR" -C "$DATA_DIR/nst_synth/" 2>/dev/null || true
        touch "$DATA_DIR/nst_synth/.done"
        echo "   OK: NST Speech Synthesis downloaded"
    fi
else
    echo "   Already downloaded"
fi

echo ""
echo "-- Downloading NB Tale (sbr-31) --"
echo "   380 speakers, 24 dialect areas, ~12 hours"

NB_TALE_URL="https://www.nb.no/sbfil/tale/nb_tale.tar.gz"
NB_TALE_TAR="$DATA_DIR/nb_tale/nb_tale.tar.gz"

if [ ! -f "$DATA_DIR/nb_tale/.done" ]; then
    echo "   Downloading NB Tale..."
    wget -q --show-progress -O "$NB_TALE_TAR" "$NB_TALE_URL" 2>&1 || {
        NB_TALE_URL_ALT="https://www.nb.no/sbfil/tale/nb_tale.zip"
        wget -q --show-progress -O "$DATA_DIR/nb_tale/nb_tale.zip" "$NB_TALE_URL_ALT" 2>&1 || {
            echo "   FAILED: Manual download from https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-31/"
        }
    }
    if [ -f "$NB_TALE_TAR" ] && [ -s "$NB_TALE_TAR" ]; then
        tar -xzf "$NB_TALE_TAR" -C "$DATA_DIR/nb_tale/" 2>/dev/null || true
        touch "$DATA_DIR/nb_tale/.done"
        echo "   OK: NB Tale downloaded"
    elif [ -f "$DATA_DIR/nb_tale/nb_tale.zip" ]; then
        cd "$DATA_DIR/nb_tale/" && unzip -o nb_tale.zip 2>/dev/null || true
        touch "$DATA_DIR/nb_tale/.done"
        echo "   OK: NB Tale downloaded"
    fi
else
    echo "   Already downloaded"
fi

echo ""
echo "-- Data inventory --"
echo "   NST Synth: $(find $DATA_DIR/nst_synth -name '*.wav' 2>/dev/null | wc -l) WAV files"
echo "   NB Tale:   $(find $DATA_DIR/nb_tale -name '*.wav' 2>/dev/null | wc -l) WAV files"
