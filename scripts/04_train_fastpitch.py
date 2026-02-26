#!/usr/bin/env python3
"""
Step 4: Fine-tune FastPitch on Norwegian using NeMo's native training API.
Uses model.setup_training_data() / setup_validation_data() + NeMo Trainer.
"""
import os, json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(SCRIPT_DIR.parent / "data")))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(SCRIPT_DIR.parent / "models")))
MANIFEST_DIR = DATA_DIR / "manifests"
SUP_DATA_DIR = DATA_DIR / "sup_data"

def main():
    print("Step 4: Fine-tune FastPitch on Norwegian")
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
        from nemo.collections.tts.models import FastPitchModel
        from nemo.utils.exp_manager import exp_manager
        from nemo.core.config import hydra_runner
        from omegaconf import OmegaConf, open_dict
        import nemo.core as nemo_core

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

        print("Loading pretrained FastPitch (English)")
        model = FastPitchModel.from_pretrained("tts_en_fastpitch")
        print("  Pretrained model loaded")

        print("Configuring for Norwegian")
        with open_dict(model.cfg):
            model.cfg.train_ds.manifest_filepath = str(train_manifest)
            model.cfg.validation_ds.manifest_filepath = str(val_manifest)
            model.cfg.train_ds.batch_size = 16
            model.cfg.validation_ds.batch_size = 8
            model.cfg.train_ds.num_workers = 4
            model.cfg.validation_ds.num_workers = 2
            model.cfg.sup_data_path = str(SUP_DATA_DIR)
            model.cfg.sup_data_types = ["align_prior_matrix", "pitch"]
            model.cfg.optim.lr = 1e-4
            model.cfg.optim.name = "adam"
            model.cfg.optim.weight_decay = 1e-6

        model.setup_training_data(model.cfg.train_ds)
        model.setup_validation_data(model.cfg.validation_ds)

        print("Starting training")
        # Use NeMo's ptl trainer
        import pytorch_lightning as pl
        trainer = pl.Trainer(
            devices=1,
            accelerator="gpu" if device == "cuda" else "cpu",
            max_steps=max_steps,
            check_val_every_n_epoch=1,
            log_every_n_steps=50,
            enable_checkpointing=True,
            logger=False,
            default_root_dir=str(MODEL_DIR / "fastpitch_logs"),
        )
        exp_cfg = {
            "exp_dir": str(MODEL_DIR / "fastpitch_logs"),
            "name": "FastPitch_Norwegian",
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

        # NeMo models need to be set as the trainer's model
        model.set_trainer(trainer)
        trainer.fit(model)

        output_path = MODEL_DIR / "norwegian_fastpitch.nemo"
        model.save_to(str(output_path))
        print(f"FastPitch saved to {output_path}")
        print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    except ImportError as e:
        print(f"NeMo not available: {e}")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__": main()
