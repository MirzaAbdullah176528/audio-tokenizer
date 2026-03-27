"""
Kaggle TPU v5e-8 environment setup.
Call setup_kaggle_tpu() as the very first thing in your notebook,
before any other JAX import or operation.
"""
import os
import sys
from typing import Optional


def setup_kaggle_tpu() -> int:
    """
    Initialize JAX for Kaggle TPU v5e-8 environment.
    Must be called before any jax.numpy or jax.device operations.

    Returns:
        num_devices: number of TPU chips available (8 on Kaggle v5e-8)
    """
    import jax

    tpu_address = os.environ.get("KAGGLE_TPU_ADDR", None)

    if tpu_address:
        jax.distributed.initialize(
            coordinator_address=f"{tpu_address}:8476",
            num_processes=1,
            process_id=0,
        )
        print(f"JAX distributed initialized. TPU address: {tpu_address}")
    else:
        try:
            jax.distributed.initialize()
            print("JAX distributed initialized (auto-detected TPU).")
        except Exception as e:
            print(f"Note: jax.distributed.initialize() skipped: {e}")
            print("Running in single-process mode.")

    devices = jax.devices()
    num_devices = len(devices)
    print(f"Available JAX devices: {num_devices}")
    for i, d in enumerate(devices):
        print(f"  [{i}] {d}")

    if num_devices < 8:
        print(
            f"WARNING: Expected 8 TPU chips, found {num_devices}. "
            "Make sure you selected TPU v5e-8 in Kaggle notebook settings."
        )

    return num_devices


def get_kaggle_paths(base_dataset_name: Optional[str] = None) -> dict:
    """
    Returns the canonical Kaggle paths for this project.

    Args:
        base_dataset_name: name of your Kaggle input dataset containing
                           Urdu audio, e.g. 'urdu-speech-corpus'

    Returns:
        dict with keys: audio_dir, checkpoint_dir, output_dir, tokenizer_dir
    """
    working = "/kaggle/working"
    input_base = "/kaggle/input"

    paths = {
        "checkpoint_dir": f"{working}/checkpoints",
        "output_dir": f"{working}/outputs",
        "tokenizer_dir": f"{working}/tokenizer",
        "log_dir": f"{working}/logs",
    }

    if base_dataset_name:
        paths["audio_dir"] = f"{input_base}/{base_dataset_name}/audio"
        paths["transcript_dir"] = f"{input_base}/{base_dataset_name}/transcripts"
        paths["pretrained_mimi"] = f"{input_base}/{base_dataset_name}/mimi_checkpoint"
        paths["qwen_weights"] = f"{input_base}/{base_dataset_name}/qwen25_weights"
    else:
        paths["audio_dir"] = f"{input_base}/audio"
        paths["transcript_dir"] = f"{input_base}/transcripts"
        paths["pretrained_mimi"] = f"{input_base}/mimi_checkpoint"
        paths["qwen_weights"] = f"{input_base}/qwen25_weights"

    for key in ["checkpoint_dir", "output_dir", "tokenizer_dir", "log_dir"]:
        os.makedirs(paths[key], exist_ok=True)

    return paths


def check_disk_space():
    """Print current /kaggle/working disk usage. Limit is ~20GB."""
    import shutil
    total, used, free = shutil.disk_usage("/kaggle/working")
    used_gb = used / (1024 ** 3)
    free_gb = free / (1024 ** 3)
    total_gb = total / (1024 ** 3)
    print(f"Disk: {used_gb:.1f}GB used / {total_gb:.1f}GB total ({free_gb:.1f}GB free)")
    if free_gb < 2.0:
        print("WARNING: Less than 2GB free. Clean old checkpoints or commit to dataset.")


def download_qwen25_to_kaggle_input(
    hf_model_id: str = "Qwen/Qwen2.5-1.5B",
    save_dir: str = "/kaggle/working/qwen25_weights",
) -> str:
    """
    Downloads Qwen2.5 weights from HuggingFace into /kaggle/working.
    Then add /kaggle/working as a dataset to reuse across sessions.

    NOTE: Kaggle notebooks have internet access during the session.
    For reliable offline re-use, commit the weights to a private Kaggle Dataset.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("pip install transformers")

    import torch
    os.makedirs(save_dir, exist_ok=True)

    print(f"Downloading {hf_model_id} → {save_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.save_pretrained(save_dir)

    tokenizer_hf = AutoTokenizer.from_pretrained(hf_model_id)
    tokenizer_hf.save_pretrained(save_dir)

    del model
    print(f"Saved to {save_dir}")
    check_disk_space()
    return save_dir


def print_kaggle_quota_reminder():
    print("""
╔══════════════════════════════════════════════════════╗
║         KAGGLE TPU v5e-8 QUOTA REMINDERS             ║
╠══════════════════════════════════════════════════════╣
║  • TPU quota:  ~20 hours / week                      ║
║  • Session cap: 12 hours max per run                 ║
║  • Working dir: 20 GB limit (/kaggle/working)        ║
║  • Save checkpoints every ~2000 steps                ║
║  • Commit /kaggle/working to Dataset after each run  ║
║    so you can resume next session                    ║
╚══════════════════════════════════════════════════════╝
""")
