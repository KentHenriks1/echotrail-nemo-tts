#!/usr/bin/env python3
"""
Step 4: Fine-tune FastPitch on Norwegian using NeMo 2.7.
Uses pre-processed IPA manifests (from step 03).
Character-level tokenizer on IPA text — no runtime G2P needed.
"""
import os, json
from pathlib import Path
from typing import List

# Ensure espeak library findable (needed by NeMo text_normalizer)
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/usr/lib/x86_64-linux-gnu/libespeak-ng.so"
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(SCRIPT_DIR.parent / "data"))).resolve()
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(SCRIPT_DIR.parent / "models"))).resolve()
MANIFEST_DIR = DATA_DIR / "manifests"
SUP_DATA_DIR = DATA_DIR / "sup_data"
NEW_DS_CLASS = "nemo.collections.tts.data.dataset.TTSDataset"


def build_vocab_from_manifest(manifest_path):
    """Scan IPA manifest to build exact character vocabulary."""
    chars = set()
    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line)
            chars.update(entry.get("text", ""))
    special = ["<pad>", "<blank>", "<oov>"]
    all_tokens = special + sorted(chars)
    return all_tokens


def create_tokenizer(vocab):
    """Create character-level tokenizer from vocabulary (no G2P needed)."""
    from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import BaseTokenizer

    class IPACharTokenizer(BaseTokenizer):
        """Character-level tokenizer for pre-processed IPA text."""
        def __init__(self, vocab_list):
            self.pad = 0
            self.blank = 1
            self.oov = 2
            self._token2id = {t: i for i, t in enumerate(vocab_list)}
            self._id2token = {i: t for i, t in enumerate(vocab_list)}
            self.tokens = vocab_list
            self.vocab_size = len(vocab_list)
            # No G2P — text is already IPA
            self.phoneme_probability = None
            self.text_preprocessing_func = lambda x: x.strip()

        def encode(self, text):
            return [self._token2id.get(c, self.oov) for c in self.text_preprocessing_func(text)]

        def decode(self, ids):
            return "".join(self._id2token.get(i, "?") for i in ids)

    return IPACharTokenizer(vocab)


def resize_embeddings(model, new_vocab_size):
    import torch, torch.nn as nn
    old_emb = model.fastpitch.encoder.word_emb
    old_vs, dim = old_emb.weight.shape
    print(f"  Resize embedding: {old_vs} -> {new_vocab_size} (dim={dim})")
    new_emb = nn.Embedding(new_vocab_size, dim, padding_idx=0)
    nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)
    with torch.no_grad():
        new_emb.weight[0].zero_()
    model.fastpitch.encoder.word_emb = new_emb
    from omegaconf import open_dict
    with open_dict(model.cfg):
        model.cfg.symbols_embedding_dim = dim
        model.cfg.n_symbols = new_vocab_size
    print(f"  Embedding resized OK")


def main():
    print("Step 4: Fine-tune FastPitch on Norwegian (pre-processed IPA)")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_manifest = MANIFEST_DIR / "norwegian_train_ipa.json"
    val_manifest = MANIFEST_DIR / "norwegian_val_ipa.json"
    if not train_manifest.exists():
        print("IPA manifest not found. Run step 03 first.")
        return

    with open(train_manifest) as f:
        entries = [json.loads(l) for l in f]
    total_dur = sum(e["duration"] for e in entries)
    print(f"  Data: {len(entries)} utts, {total_dur/60:.1f} min")
    max_steps = max(3000, min(10000, int(total_dur / 60 * 100)))
    print(f"  Max steps: {max_steps}")

    try:
        import torch
        import lightning.pytorch as pl
        from omegaconf import OmegaConf, open_dict
        from nemo.collections.tts.models import FastPitchModel
        from nemo.utils.exp_manager import exp_manager

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

        # 1. Build vocab from IPA manifest
        print("Building IPA vocabulary from manifest")
        vocab = build_vocab_from_manifest(str(train_manifest))
        print(f"  Vocab: {len(vocab)} tokens")

        # 2. Create char-level tokenizer (no runtime G2P)
        tokenizer = create_tokenizer(vocab)
        test_ids = tokenizer.encode("hˈɑɪ dˌɛtːɑ")
        print(f"  Test encode: {len(test_ids)} tokens, oov={sum(1 for t in test_ids if t == 2)}")

        # 3. Load pretrained model
        print("Loading pretrained FastPitch")
        model = FastPitchModel.from_pretrained("tts_en_fastpitch")

        # 4. Patch model
        print("Patching model for Norwegian IPA")
        model.vocab = tokenizer
        model.ds_class = NEW_DS_CLASS
        resize_embeddings(model, tokenizer.vocab_size)

        abs_train = str(train_manifest)
        abs_val = str(val_manifest)
        abs_sup = str(SUP_DATA_DIR)

        with open_dict(model.cfg):
            model.cfg.train_ds.dataset._target_ = NEW_DS_CLASS
            model.cfg.train_ds.dataset.manifest_filepath = abs_train
            model.cfg.train_ds.dataset.sup_data_path = abs_sup
            model.cfg.train_ds.manifest_filepath = abs_train
            model.cfg.validation_ds.dataset._target_ = NEW_DS_CLASS
            model.cfg.validation_ds.dataset.manifest_filepath = abs_val
            model.cfg.validation_ds.dataset.sup_data_path = abs_sup
            model.cfg.validation_ds.manifest_filepath = abs_val
            model.cfg.train_ds.batch_size = 16
            model.cfg.validation_ds.batch_size = 8
            model.cfg.train_ds.dataloader_params.num_workers = 0
            model.cfg.validation_ds.dataloader_params.num_workers = 0
            model.cfg.sup_data_path = abs_sup
            model.cfg.sup_data_types = ["align_prior_matrix", "pitch"]
            model.cfg.optim.lr = 1e-4
            model.cfg.optim.name = "adam"
            model.cfg.optim.weight_decay = 1e-6

        # 5. Setup data (IPA text already in manifest — no G2P needed)
        print("Setup data loaders (IPA pre-processed)")
        model.setup_training_data(model.cfg.train_ds)
        model.setup_validation_data(model.cfg.validation_ds)

        # 6. Train
        print(f"Training ({max_steps} steps)")
        trainer = pl.Trainer(
            devices=1, accelerator="gpu" if device == "cuda" else "cpu",
            max_steps=max_steps, check_val_every_n_epoch=1,
            log_every_n_steps=50, enable_checkpointing=False,
            logger=False, default_root_dir=str(MODEL_DIR / "fastpitch_logs"),
        )
        exp_cfg = {
            "exp_dir": str(MODEL_DIR / "fastpitch_logs"),
            "name": "FastPitch_Norwegian_IPA",
            "create_tensorboard_logger": True,
            "create_wandb_logger": False,
            "checkpoint_callback_params": {
                "monitor": "val_loss", "mode": "min",
                "save_top_k": 3, "save_last": True,
            },
        }
        exp_manager(trainer, OmegaConf.create(exp_cfg))
        model.set_trainer(trainer)
        trainer.fit(model)

        # 7. Save
        output_path = MODEL_DIR / "norwegian_fastpitch.nemo"
        model.save_to(str(output_path))
        print(f"Saved: {output_path} ({output_path.stat().st_size/1024/1024:.1f} MB)")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
