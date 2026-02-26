#!/usr/bin/env python3
"""
Step 4: Fine-tune FastPitch on Norwegian using NeMo 2.7.
- Uses espeak-ng + phonemizer for Norwegian IPA G2P
- Custom NorwegianG2P class wrapping phonemizer
- IPATokenizer with Norwegian locale support
- Embedding layer resize for new vocabulary
- lightning.pytorch Trainer for NeMo 2.7 compat
"""
import os, sys, json, re, string
from pathlib import Path
from typing import List, Optional, Callable, Union

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(SCRIPT_DIR.parent / "data"))).resolve()
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(SCRIPT_DIR.parent / "models"))).resolve()
MANIFEST_DIR = DATA_DIR / "manifests"
SUP_DATA_DIR = DATA_DIR / "sup_data"

NEW_DS_CLASS = "nemo.collections.tts.data.dataset.TTSDataset"


class NorwegianG2P:
    """Norwegian grapheme-to-phoneme using espeak-ng via phonemizer."""

    def __init__(self, phoneme_probability: Optional[float] = None):
        from phonemizer import phonemize
        from phonemizer.backend import EspeakBackend
        self.phonemize = phonemize
        self.phoneme_probability = phoneme_probability
        # Verify Norwegian is available
        self._backend = EspeakBackend('nb', language_switch='remove-flags')
        print("  NorwegianG2P initialized with espeak-ng 'nb' backend")

    def __call__(self, text: str) -> List[str]:
        """Convert Norwegian text to IPA phoneme list."""
        # Phonemize the text
        ipa = self.phonemize(
            text,
            language='nb',
            backend='espeak',
            strip=True,
            preserve_punctuation=True,
            with_stress=True,
        )
        # Split into individual phoneme tokens
        # IPA output uses spaces between words - we preserve word boundaries
        tokens = []
        for char in ipa:
            if char == ' ':
                tokens.append(' ')
            elif char in string.punctuation or char in '.,!?;:':
                tokens.append(char)
            else:
                tokens.append(char)
        return tokens


def build_norwegian_ipa_vocab():
    """Build IPA vocabulary from Norwegian phoneme set."""
    # Core Norwegian IPA vowels
    vowels = list("iɪeɛæaɑɔoʊuʉyøœəɐ")
    # Long vowel markers
    long_markers = ["ː"]
    # Norwegian consonants
    consonants = list("pbtdkɡmnŋfvsʃʂçxhrlɽjwɾ")
    # Additional IPA symbols used in Norwegian
    extras = ["ˈ", "ˌ", "ʰ", "ʲ", "ˑ", "̃", "̥"]
    # Standard punctuation
    punct = list(".,!?;:-–'\"()[] ")
    # Digits (for numbers in text)
    digits = list("0123456789")
    # Special tokens
    special = ["<pad>", "<blank>", "<oov>"]

    all_tokens = special + sorted(set(
        vowels + long_markers + consonants + extras + punct + digits
    ))
    return all_tokens


def create_norwegian_tokenizer():
    """Create a tokenizer for Norwegian IPA phonemes."""
    from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import BaseTokenizer

    class NorwegianIPATokenizer(BaseTokenizer):
        """IPA tokenizer for Norwegian using espeak-ng."""

        PAD = "<pad>"
        BLANK = "<blank>"
        OOV = "<oov>"

        def __init__(self, g2p=None):
            # Build vocabulary
            vocab = build_norwegian_ipa_vocab()
            self.pad = vocab.index(self.PAD)
            self.blank = vocab.index(self.BLANK) if self.BLANK in vocab else None
            self.oov = vocab.index(self.OOV)

            # Token <-> ID mappings
            self._token2id = {t: i for i, t in enumerate(vocab)}
            self._id2token = {i: t for i, t in enumerate(vocab)}
            self.tokens = vocab
            self.vocab_size = len(vocab)

            self.g2p = g2p or NorwegianG2P()
            self.phoneme_probability = getattr(g2p, 'phoneme_probability', None)
            self.text_preprocessing_func = lambda x: x.strip()
            print(f"  NorwegianIPATokenizer: {self.vocab_size} tokens")

        def encode(self, text: str) -> List[int]:
            """Encode text to token IDs via IPA."""
            text = self.text_preprocessing_func(text)
            phonemes = self.g2p(text)
            ids = []
            for p in phonemes:
                if p in self._token2id:
                    ids.append(self._token2id[p])
                else:
                    ids.append(self.oov)
            return ids

        def decode(self, ids: List[int]) -> str:
            return "".join(self._id2token.get(i, "?") for i in ids)

    return NorwegianIPATokenizer()


def resize_embeddings(model, new_vocab_size):
    """Resize FastPitch text encoder embedding layer for new vocabulary."""
    import torch
    import torch.nn as nn

    # Find the text embedding layer
    old_emb = model.fastpitch.encoder.word_emb
    old_vocab_size, emb_dim = old_emb.weight.shape
    print(f"  Resizing embeddings: {old_vocab_size} -> {new_vocab_size} (dim={emb_dim})")

    if new_vocab_size == old_vocab_size:
        print("  No resize needed")
        return

    # Create new embedding layer
    new_emb = nn.Embedding(new_vocab_size, emb_dim, padding_idx=old_emb.padding_idx)

    # Initialize with small random values
    nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)

    # Copy over existing weights where possible
    copy_size = min(old_vocab_size, new_vocab_size)
    with torch.no_grad():
        new_emb.weight[:copy_size] = old_emb.weight[:copy_size]

    # Replace the embedding layer
    model.fastpitch.encoder.word_emb = new_emb
    print(f"  Embedding resized successfully")

    # Also update the model config
    from omegaconf import open_dict
    with open_dict(model.cfg):
        model.cfg.symbols_embedding_dim = emb_dim
        model.cfg.n_symbols = new_vocab_size


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
    print(f"  Training data: {len(entries)} utterances, {total_dur/60:.1f} min")
    max_steps = max(3000, min(10000, int(total_dur / 60 * 100)))
    print(f"  Training steps: {max_steps}")

    try:
        import torch
        import lightning.pytorch as pl
        from omegaconf import OmegaConf, open_dict
        from nemo.collections.tts.models import FastPitchModel
        from nemo.utils.exp_manager import exp_manager

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

        # 1. Create Norwegian tokenizer
        print("Creating Norwegian IPA tokenizer")
        g2p = NorwegianG2P()
        tokenizer = create_norwegian_tokenizer()
        print(f"  Vocab size: {tokenizer.vocab_size}")

        # Test tokenizer
        test_text = "Hei dette er en test"
        test_ids = tokenizer.encode(test_text)
        print(f"  Test: '{test_text}' -> {len(test_ids)} tokens")

        # 2. Load pretrained model
        print("Loading pretrained FastPitch (English)")
        model = FastPitchModel.from_pretrained("tts_en_fastpitch")

        # 3. Replace tokenizer and resize embeddings
        print("Replacing tokenizer with Norwegian IPA")
        model.vocab = tokenizer
        model.ds_class = NEW_DS_CLASS
        resize_embeddings(model, tokenizer.vocab_size)

        # 4. Configure for Norwegian data
        print("Configuring for Norwegian data")
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
            model.cfg.train_ds.dataloader_params.num_workers = 4
            model.cfg.validation_ds.dataloader_params.num_workers = 2
            model.cfg.sup_data_path = abs_sup
            model.cfg.sup_data_types = ["align_prior_matrix", "pitch"]
            model.cfg.optim.lr = 1e-4
            model.cfg.optim.name = "adam"
            model.cfg.optim.weight_decay = 1e-6

        # 5. Setup data loaders
        print("Setting up data loaders")
        model.setup_training_data(model.cfg.train_ds)
        model.setup_validation_data(model.cfg.validation_ds)
        print("  Data loaders configured")

        # 6. Create trainer
        print(f"Starting training ({max_steps} steps)")
        trainer = pl.Trainer(
            devices=1,
            accelerator="gpu" if device == "cuda" else "cpu",
            max_steps=max_steps,
            check_val_every_n_epoch=1,
            log_every_n_steps=50,
            enable_checkpointing=False,
            logger=False,
            default_root_dir=str(MODEL_DIR / "fastpitch_logs"),
        )
        exp_cfg = {
            "exp_dir": str(MODEL_DIR / "fastpitch_logs"),
            "name": "FastPitch_Norwegian_IPA",
            "create_tensorboard_logger": True,
            "create_wandb_logger": False,
            "checkpoint_callback_params": {
                "monitor": "val_loss",
                "mode": "min",
                "save_top_k": 3,
                "save_last": True,
            },
        }
        exp_manager(trainer, OmegaConf.create(exp_cfg))
        model.set_trainer(trainer)
        trainer.fit(model)

        # 7. Save model
        output_path = MODEL_DIR / "norwegian_fastpitch.nemo"
        model.save_to(str(output_path))
        print(f"FastPitch saved to {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    except ImportError as e:
        print(f"NeMo not available: {e}")
        import traceback; traceback.print_exc()
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
