#!/usr/bin/env python3
"""
Step 4: Fine-tune FastPitch on Norwegian using NeMo 2.7.
Uses espeak-ng + phonemizer for Norwegian IPA G2P.
Resizes embedding layer for new IPA vocabulary.
"""
import os, json, string
from pathlib import Path
from typing import List, Optional

# Ensure espeak library is findable by phonemizer in all subprocesses
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")
# Also set PHONEMIZER_ESPEAK_LIBRARY directly
_espeak_lib = "/usr/lib/x86_64-linux-gnu/libespeak-ng.so"
if os.path.exists(_espeak_lib):
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = _espeak_lib

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(SCRIPT_DIR.parent / "data"))).resolve()
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(SCRIPT_DIR.parent / "models"))).resolve()
MANIFEST_DIR = DATA_DIR / "manifests"
SUP_DATA_DIR = DATA_DIR / "sup_data"
NEW_DS_CLASS = "nemo.collections.tts.data.dataset.TTSDataset"


class NorwegianG2P:
    def __init__(self, phoneme_probability=None):
        from phonemizer import phonemize
        from phonemizer.backend import EspeakBackend
        self.phonemize = phonemize
        self.phoneme_probability = phoneme_probability
        self._backend = EspeakBackend('nb', language_switch='remove-flags')
        print("  NorwegianG2P: espeak-ng 'nb' ready")

    def __call__(self, text):
        ipa = self.phonemize(text, language='nb', backend='espeak',
                             strip=True, preserve_punctuation=True, with_stress=True)
        return list(ipa)


def build_vocab():
    ipa_chars = (
        " .,!?;:-'\"()[]0123456789"
        "aAbcCdDeEfghiIjklmnNoOpqrsStuvwxyz"
        "\u00e5\u00e6\u00f8\u00c5\u00c6\u00d8"
        "\u0250\u0251\u0252\u0253\u0254\u0255\u0256\u0258\u0259\u025a\u025b\u025c\u025e"
        "\u025f\u0260\u0261\u0263\u0264\u0265\u0266\u0267\u0268\u026a\u026b\u026c\u026d"
        "\u026e\u026f\u0270\u0271\u0272\u0273\u0274\u0275\u0276\u0278\u0279\u027a\u027b"
        "\u027d\u027e\u0280\u0281\u0282\u0283\u0284\u0288\u0289\u028a\u028b\u028c\u028d"
        "\u028e\u028f\u0290\u0291\u0292\u0294\u0295\u0298\u0299\u029b\u029c\u029d\u029f"
        "\u02a1\u02a2\u02b0\u02b2\u02c8\u02cc\u02d0\u02d1\u02de\u02e0\u02e4"
        "\u0303\u0306\u0308\u030b\u030c\u030f\u0318\u0319\u031a\u031c\u031d\u031e\u031f"
        "\u0320\u0324\u0325\u032a\u032c\u032f\u0330\u0334\u0339\u033a\u033b\u033c\u033d"
        "\u0361"
        "\u03b2\u03b8\u03c7"
        "\u00e7\u00f0\u0153"
    )
    special = ["<pad>", "<blank>", "<oov>"]
    all_tokens = special + sorted(set(ipa_chars))
    return all_tokens


def create_tokenizer():
    from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import BaseTokenizer

    class NorwegianIPATokenizer(BaseTokenizer):
        PAD, BLANK, OOV = "<pad>", "<blank>", "<oov>"
        def __init__(self, g2p=None):
            vocab = build_vocab()
            self.pad = 0
            self.blank = 1
            self.oov = 2
            self._token2id = {t: i for i, t in enumerate(vocab)}
            self._id2token = {i: t for i, t in enumerate(vocab)}
            self.tokens = vocab
            self.vocab_size = len(vocab)
            self.g2p = g2p or NorwegianG2P()
            self.phoneme_probability = getattr(g2p, 'phoneme_probability', None)
            self.text_preprocessing_func = lambda x: x.strip()
            print(f"  NorwegianIPATokenizer: {self.vocab_size} tokens")

        def encode(self, text):
            phonemes = self.g2p(self.text_preprocessing_func(text))
            return [self._token2id.get(p, self.oov) for p in phonemes]

        def decode(self, ids):
            return "".join(self._id2token.get(i, "?") for i in ids)

    return NorwegianIPATokenizer()


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
    print("Step 4: Fine-tune FastPitch on Norwegian (IPA)")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    SUP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_manifest = MANIFEST_DIR / "norwegian_train.json"
    val_manifest = MANIFEST_DIR / "norwegian_val.json"
    if not train_manifest.exists():
        print("Train manifest not found. Run steps 1-2 first.")
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

        print("Creating Norwegian IPA tokenizer")
        tokenizer = create_tokenizer()

        test_ids = tokenizer.encode("Hei dette er en test")
        print(f"  Test encode: {len(test_ids)} tokens")

        print("Loading pretrained FastPitch")
        model = FastPitchModel.from_pretrained("tts_en_fastpitch")

        print("Patching model for Norwegian")
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
            # Use num_workers=0 to avoid subprocess espeak library issues
            model.cfg.train_ds.dataloader_params.num_workers = 0
            model.cfg.validation_ds.dataloader_params.num_workers = 0
            model.cfg.sup_data_path = abs_sup
            model.cfg.sup_data_types = ["align_prior_matrix", "pitch"]
            model.cfg.optim.lr = 1e-4
            model.cfg.optim.name = "adam"
            model.cfg.optim.weight_decay = 1e-6

        print("Setup data loaders")
        model.setup_training_data(model.cfg.train_ds)
        model.setup_validation_data(model.cfg.validation_ds)

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

        output_path = MODEL_DIR / "norwegian_fastpitch.nemo"
        model.save_to(str(output_path))
        print(f"Saved: {output_path} ({output_path.stat().st_size/1024/1024:.1f} MB)")

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
