#!/usr/bin/env python3
"""
Step 3b: Pre-compute ALL supplementary data (pitch, mel, alignment priors)
to disk before training. This eliminates the CPU bottleneck during training
so GPU can run at full utilization.

Run this ONCE before training. Results are cached in data/sup_data/.
"""
import os, json, sys, time
import numpy as np
from pathlib import Path

os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/usr/lib/x86_64-linux-gnu/libespeak-ng.so"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(SCRIPT_DIR.parent / "data"))).resolve()
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(SCRIPT_DIR.parent / "models"))).resolve()
MANIFEST_DIR = DATA_DIR / "manifests"
SUP_DATA_DIR = DATA_DIR / "sup_data"


def main():
    print("Step 3b: Pre-compute supplementary data for FastPitch training")
    SUP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_manifest = MANIFEST_DIR / "norwegian_train_ipa.json"
    val_manifest = MANIFEST_DIR / "norwegian_val_ipa.json"

    if not train_manifest.exists():
        print("IPA manifest not found. Run step 03 first.")
        return

    import torch
    from nemo.collections.tts.models import FastPitchModel
    from nemo.collections.tts.data.dataset import TTSDataset
    from omegaconf import open_dict
    from tqdm import tqdm

    # Build vocab and tokenizer (same as training script)
    def build_vocab(manifest_path):
        chars = set()
        with open(manifest_path) as f:
            for line in f:
                chars.update(json.loads(line).get("text", ""))
        return ["<pad>", "<blank>", "<oov>"] + sorted(chars)

    from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import BaseTokenizer

    class IPACharTokenizer(BaseTokenizer):
        def __init__(self, vocab_list):
            self.pad, self.blank, self.oov = 0, 1, 2
            self._token2id = {t: i for i, t in enumerate(vocab_list)}
            self._id2token = {i: t for i, t in enumerate(vocab_list)}
            self.tokens = vocab_list
            self.vocab_size = len(vocab_list)
            self.phoneme_probability = None
            self.text_preprocessing_func = lambda x: x.strip()
        def encode(self, text):
            return [self._token2id.get(c, self.oov) for c in self.text_preprocessing_func(text)]
        def decode(self, ids):
            return "".join(self._id2token.get(i, "?") for i in ids)

    vocab = build_vocab(str(train_manifest))
    tokenizer = IPACharTokenizer(vocab)
    print(f"  Vocab: {tokenizer.vocab_size} tokens")

    # Load model to get normalizer and config
    print("Loading pretrained FastPitch for config...")
    model = FastPitchModel.from_pretrained("tts_en_fastpitch")

    # Create dataset with sup_data generation enabled
    for manifest_path, name in [(train_manifest, "train"), (val_manifest, "val")]:
        print(f"\nPre-computing {name} supplementary data...")
        with open(manifest_path) as f:
            n_entries = sum(1 for _ in f)
        print(f"  {n_entries} entries")

        # Create TTSDataset which generates sup_data on first access
        ds_cfg = model.cfg.train_ds.dataset.copy()
        with open_dict(ds_cfg):
            ds_cfg.manifest_filepath = str(manifest_path)
            ds_cfg.sup_data_path = str(SUP_DATA_DIR)
            ds_cfg._target_ = "nemo.collections.tts.data.dataset.TTSDataset"

        from hydra.utils import instantiate
        dataset = instantiate(
            ds_cfg,
            text_normalizer=model.normalizer,
            text_normalizer_call_kwargs=model.text_normalizer_call_kwargs,
            text_tokenizer=tokenizer,
        )

        # Iterate through entire dataset to trigger sup_data generation
        start = time.time()
        errors = 0
        for i in tqdm(range(len(dataset)), desc=f"  {name}"):
            try:
                _ = dataset[i]
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"    Error at {i}: {e}")
        elapsed = time.time() - start
        print(f"  Done in {elapsed/60:.1f} min ({errors} errors)")

    # Check what was generated
    sup_files = list(SUP_DATA_DIR.rglob("*"))
    total_size = sum(f.stat().st_size for f in sup_files if f.is_file())
    print(f"\nSupplementary data: {len(sup_files)} files, {total_size/1e9:.1f} GB")
    print(f"Location: {SUP_DATA_DIR}")
    print("Ready for training — GPU will now run at full speed!")


if __name__ == "__main__":
    main()
