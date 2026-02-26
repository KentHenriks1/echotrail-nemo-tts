#!/usr/bin/env python3
"""
Step 4: Fine-tune FastPitch on Norwegian using NeMo 2.7.
Uses pre-processed IPA manifests (from step 03).
Maxes out CPU+GPU: 32 workers, prefetch, bf16, batch=64.
Epoch 0 caches sup_data, epoch 1+ runs at full GPU speed.
"""
import os, json
from pathlib import Path

os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "/usr/lib/x86_64-linux-gnu/libespeak-ng.so"
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(SCRIPT_DIR.parent / "data"))).resolve()
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(SCRIPT_DIR.parent / "models"))).resolve()
MANIFEST_DIR = DATA_DIR / "manifests"
SUP_DATA_DIR = DATA_DIR / "sup_data"
NEW_DS_CLASS = "nemo.collections.tts.data.dataset.TTSDataset"


def build_vocab_from_manifest(manifest_path):
    chars = set()
    with open(manifest_path) as f:
        for line in f:
            chars.update(json.loads(line).get("text", ""))
    return ["<pad>", "<blank>", "<oov>"] + sorted(chars)


def create_tokenizer(vocab):
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


def main():
    print("Step 4: FastPitch Norwegian (IPA) — max throughput")
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
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9 if device == "cuda" else 0
        n_cpus = os.cpu_count() or 8
        # Use many workers but cap at reasonable level
        workers = min(32, max(8, n_cpus // 4))
        batch_size = 64 if gpu_mem >= 70 else (32 if gpu_mem >= 30 else 16)
        val_batch = batch_size // 2
        print(f"  Device: {device} ({gpu_mem:.0f} GB), CPUs: {n_cpus}, workers: {workers}, batch: {batch_size}")

        vocab = build_vocab_from_manifest(str(train_manifest))
        tokenizer = create_tokenizer(vocab)
        print(f"  Vocab: {tokenizer.vocab_size} tokens")

        print("Loading pretrained FastPitch")
        model = FastPitchModel.from_pretrained("tts_en_fastpitch")
        model.vocab = tokenizer
        model.ds_class = NEW_DS_CLASS
        resize_embeddings(model, tokenizer.vocab_size)

        abs_train, abs_val, abs_sup = str(train_manifest), str(val_manifest), str(SUP_DATA_DIR)

        with open_dict(model.cfg):
            model.cfg.train_ds.dataset._target_ = NEW_DS_CLASS
            model.cfg.train_ds.dataset.manifest_filepath = abs_train
            model.cfg.train_ds.dataset.sup_data_path = abs_sup
            model.cfg.train_ds.manifest_filepath = abs_train
            model.cfg.validation_ds.dataset._target_ = NEW_DS_CLASS
            model.cfg.validation_ds.dataset.manifest_filepath = abs_val
            model.cfg.validation_ds.dataset.sup_data_path = abs_sup
            model.cfg.validation_ds.manifest_filepath = abs_val
            model.cfg.train_ds.batch_size = batch_size
            model.cfg.validation_ds.batch_size = val_batch
            # Max CPU utilization: many workers + prefetch
            model.cfg.train_ds.dataloader_params.num_workers = workers
            model.cfg.train_ds.dataloader_params.pin_memory = True
            model.cfg.train_ds.dataloader_params.persistent_workers = True
            model.cfg.train_ds.dataloader_params.prefetch_factor = 4
            model.cfg.validation_ds.dataloader_params.num_workers = workers
            model.cfg.validation_ds.dataloader_params.pin_memory = True
            model.cfg.validation_ds.dataloader_params.persistent_workers = True
            model.cfg.validation_ds.dataloader_params.prefetch_factor = 4
            model.cfg.sup_data_path = abs_sup
            model.cfg.sup_data_types = ["align_prior_matrix", "pitch"]
            model.cfg.optim.lr = 2e-4
            model.cfg.optim.name = "adam"
            model.cfg.optim.weight_decay = 1e-6

        print("Setup data loaders")
        model.setup_training_data(model.cfg.train_ds)
        model.setup_validation_data(model.cfg.validation_ds)

        print(f"Training ({max_steps} steps, batch={batch_size}, bf16, workers={workers})")
        trainer = pl.Trainer(
            devices=1,
            accelerator="gpu" if device == "cuda" else "cpu",
            max_steps=max_steps,
            check_val_every_n_epoch=1,
            log_every_n_steps=50,
            enable_checkpointing=False,
            logger=False,
            precision="bf16-mixed",
            default_root_dir=str(MODEL_DIR / "fastpitch_logs"),
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
