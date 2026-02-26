#!/usr/bin/env python3
import os, json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(SCRIPT_DIR.parent / "data")))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(SCRIPT_DIR.parent / "models")))
MANIFEST_DIR = DATA_DIR / "manifests"

def main():
    print("Step 5: Fine-tune HiFi-GAN Vocoder")
    train_manifest = MANIFEST_DIR / "norwegian_train.json"
    val_manifest = MANIFEST_DIR / "norwegian_val.json"
    fastpitch_path = MODEL_DIR / "norwegian_fastpitch.nemo"
    if not train_manifest.exists():
        print("Train manifest not found. Run steps 1-2 first.")
        return
    try:
        import torch, numpy as np
        import pytorch_lightning as pl
        from nemo.collections.tts.models import HifiGanModel, FastPitchModel
        from nemo.utils.exp_manager import exp_manager
        from omegaconf import OmegaConf, open_dict
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")
        print("Generating mels from trained FastPitch")
        if fastpitch_path.exists():
            fp_model = FastPitchModel.restore_from(str(fastpitch_path))
            print("  Loaded fine-tuned Norwegian FastPitch")
        else:
            print("  No fine-tuned FastPitch found, using pretrained English")
            fp_model = FastPitchModel.from_pretrained("tts_en_fastpitch")
        fp_model = fp_model.to(device).eval()
        mel_dir = DATA_DIR / "generated_mels"
        mel_dir.mkdir(exist_ok=True)
        with open(train_manifest) as f:
            entries = [json.loads(l) for l in f]
        mel_manifest = []
        print(f"  Generating mels for {len(entries)} utterances...")
        for i, entry in enumerate(entries):
            try:
                with torch.no_grad():
                    parsed = fp_model.parse(entry["text"])
                    spec = fp_model.generate_spectrogram(tokens=parsed)
                mel_path = mel_dir / f"mel_{i:06d}.npy"
                np.save(str(mel_path), spec.cpu().numpy())
                mel_manifest.append({"audio_filepath": entry["audio_filepath"], "mel_filepath": str(mel_path), "text": entry["text"], "duration": entry["duration"]})
                if (i+1) % 200 == 0: print(f"    {i+1}/{len(entries)}")
            except Exception as e:
                if i < 3: print(f"    Skipped {i}: {e}")
        print(f"  Generated {len(mel_manifest)} mels")
        del fp_model; torch.cuda.empty_cache()
        print("Loading pretrained HiFi-GAN")
        hifigan = HifiGanModel.from_pretrained("tts_en_hifigan")
        cfg = hifigan.cfg.copy()
        with open_dict(cfg):
            cfg.train_ds.manifest_filepath = str(train_manifest)
            cfg.validation_ds.manifest_filepath = str(val_manifest)
            cfg.train_ds.batch_size = 16
            cfg.validation_ds.batch_size = 8
            cfg.optim.lr = 2e-4
        hifigan.cfg = cfg
        max_steps = max(2000, min(5000, len(mel_manifest) * 2))
        print(f"  Training steps: {max_steps}")
        trainer = pl.Trainer(devices=1, accelerator="gpu" if device == "cuda" else "cpu", max_steps=max_steps, check_val_every_n_epoch=1, log_every_n_steps=50, enable_checkpointing=True, default_root_dir=str(MODEL_DIR / "hifigan_logs"))
        exp_cfg = {"exp_dir": str(MODEL_DIR / "hifigan_logs"), "name": "HiFiGAN_Norwegian", "create_tensorboard_logger": True, "create_wandb_logger": False, "checkpoint_callback_params": {"monitor": "val_loss", "mode": "min", "save_top_k": 2, "save_last": True}}
        exp_manager(trainer, OmegaConf.create(exp_cfg))
        print("Starting HiFi-GAN training")
        trainer.fit(hifigan)
        output_path = MODEL_DIR / "norwegian_hifigan.nemo"
        hifigan.save_to(str(output_path))
        print(f"HiFi-GAN saved to {output_path} ({output_path.stat().st_size/1024/1024:.1f} MB)")
    except ImportError as e:
        print(f"NeMo not available: {e}")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__": main()
