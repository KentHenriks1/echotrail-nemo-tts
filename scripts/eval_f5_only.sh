#!/bin/bash
# Test F5-TTS Norwegian separately
set -e

echo "=== F5-TTS Norwegian eval ==="
python3 << 'PYEOF'
import torch, os, soundfile as sf, glob, time
os.makedirs('/workspace/tts_eval', exist_ok=True)

# Check F5-TTS API
try:
    import f5_tts
    print(f"f5_tts version: {f5_tts.__version__ if hasattr(f5_tts, '__version__') else 'unknown'}")
except:
    pass

# Try different API approaches
try:
    # Approach 1: newer API without model_type
    from f5_tts.api import F5TTS
    import inspect
    print(f"F5TTS.__init__ params: {inspect.signature(F5TTS.__init__)}")

    # Try without model_type
    model = F5TTS(ckpt_file='hf://akhbar/F5_Norwegian', device='cuda')
    print("Model loaded (approach 1)!")
except Exception as e1:
    print(f"Approach 1 failed: {e1}")
    try:
        # Approach 2: use load_model directly
        from f5_tts.api import F5TTS
        model = F5TTS(device='cuda')
        # Then load custom checkpoint
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(repo_id='akhbar/F5_Norwegian', filename='model_1200000.safetensors')
        print(f"Downloaded checkpoint: {ckpt}")
        model = F5TTS(ckpt_file=ckpt, device='cuda')
        print("Model loaded (approach 2)!")
    except Exception as e2:
        print(f"Approach 2 failed: {e2}")
        try:
            # Approach 3: use infer_cli or different import
            from f5_tts.infer.utils_infer import load_model, infer_process
            print("Trying utils_infer approach...")
            from huggingface_hub import hf_hub_download
            ckpt = hf_hub_download(repo_id='akhbar/F5_Norwegian', filename='model_1200000.safetensors')
            print(f"Checkpoint: {ckpt}")
            # This approach needs more investigation
            print("Need to check F5-TTS docs for correct loading")
        except Exception as e3:
            print(f"Approach 3 failed: {e3}")
            import traceback; traceback.print_exc()
            exit(1)

# If we got a model, run inference
try:
    ref_wavs = sorted(glob.glob('/workspace/echotrail-nemo-tts/data/wavs/*.wav'))
    ref_audio = ref_wavs[0] if ref_wavs else None
    print(f"Reference audio: {ref_audio}")

    if ref_audio and model:
        texts = [
            'Hei, jeg heter EchoTrail og jeg kan hjelpe deg med å finne veien.',
            'Det er viktig å ta vare på naturen rundt oss.',
            'Godmorgen! Hvordan har du det i dag?',
            'Norge er et vakkert land med fjorder, fjell og nordlys.',
            'Denne setningen tester om modellen håndterer lengre tekst med flere ord og naturlig prosodi.',
        ]
        for i, text in enumerate(texts):
            print(f'[{i+1}/5] Text: {text}')
            t0 = time.time()
            wav, sr, _ = model.infer(ref_file=ref_audio, ref_text='', gen_text=text)
            elapsed = time.time() - t0
            sf.write(f'/workspace/tts_eval/f5tts_{i+1}.wav', wav, sr)
            print(f'  Time: {elapsed:.3f}s  Saved: f5tts_{i+1}.wav\n')
        print('=== F5-TTS DONE ===')
except Exception as e:
    print(f"Inference error: {e}")
    import traceback; traceback.print_exc()

print("\nFiles:")
os.system("ls -la /workspace/tts_eval/")
PYEOF
