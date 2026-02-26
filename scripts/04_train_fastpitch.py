#!/usr/bin/env python3
"""
Step 4: Fine-tune FastPitch on Norwegian using NeMo 2.7.
Fixes _target_ path for TTSDataset (moved in NeMo 2.x).
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
        import pytorch_lightning as pl
        from nemo.collections.tts.models import FastPitchModel
        from nemo.utils.exp_manager import exp_manager
        from omegaconf import OmegaConf, open_dict

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")

        print("Loading pretrained FastPitch (English)")
        model = FastPitchModel.from_pretrained("tts_en_fastpitch")
        print("  Pretrained model loaded")

        print("Configuring for Norwegian")
        with open_dict(model.cfg):
            # Fix dataset _target_ for NeMo 2.x (moved from .torch.data to .data.dataset)
            if hasattr(model.cfg, 'train_ds') and hasattr(model.cfg.train_ds, 'dataset'):
                if hasattr(model.cfg.train_ds.dataset, '_target_'):
                    old_target = model.cfg.train_ds.dataset._target_
                    if 'torch.data' in old_target:
                        new_target = old_target.replace('torch.data', 'data.dataset')
                        model.cfg.train_ds.dataset._target_ = new_target
                        print(f"  Fixed train dataset target: {new_target}")
            if hasattr(model.cfg, 'validation_ds') and hasattr(model.cfg.validation_ds, 'dataset'):
                if hasattr(model.cfg.validation_ds.dataset, '_target_'):
                    old_target = model.cfg.validation_ds.dataset._target_
                    if 'torch.data' in old_target:
                        new_target = old_target.replace('torch.data', 'data.dataset')
                        model.cfg.validation_ds.dataset._target_ = new_target
                        print(f"  Fixed val dataset target: {new_target}")

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
        print("  Data loaders configured")

        print("Starting training")
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
