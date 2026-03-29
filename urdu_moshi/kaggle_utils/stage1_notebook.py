# ── CELL 0: Download all assets from HuggingFace ─────────────────
import os
import subprocess

HF_TOKEN = "YOUR_HF_TOKEN_HERE"  # paste your HuggingFace read token here
subprocess.run(["pip", "install", "-q", "huggingface_hub"], check=True)
from huggingface_hub import snapshot_download

# 1. Urdu audio (batch_00001 only — enough for full Stage 1 session)
AUDIO_DIR = "/kaggle/working/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)
print("Downloading Urdu audio batch_00001...")
snapshot_download(
    repo_id="mirza176528470100/audio_dataset",
    repo_type="dataset",
    local_dir=AUDIO_DIR,
    token=HF_TOKEN,
    allow_patterns=["batch_00001/**"],
    ignore_patterns=["*.json", "*.md"],
)
print("Audio ready.")

# 2. Mimi pretrained weights
print("Downloading Mimi weights...")
snapshot_download(
    repo_id="kyutai/moshika-pytorch-bf16",
    repo_type="model",
    local_dir="/kaggle/working/mimi_weights",
    ignore_patterns=["*.md", "*.txt"],
)
print("Mimi weights ready.")

# 3. Qwen2.5-1.5B weights
print("Downloading Qwen2.5-1.5B weights...")
snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B",
    repo_type="model",
    local_dir="/kaggle/working/qwen25_weights",
    ignore_patterns=["*.md", "*.txt"],
)
print("Qwen2.5 weights ready.")


# ================================================================
# URDU MOSHI — STAGE 1 TRAINING NOTEBOOK
# TPU v5e-8 on Kaggle
#
# HOW TO USE:
#   1. Create a Kaggle Notebook
#   2. Accelerator → TPU VM v5e-8
#   3. Add urdu-moshi-code as input dataset
#   4. Paste this entire file or upload as .ipynb
#   5. Run all cells
# ================================================================


# ── CELL 1: Install dependencies ─────────────────────────────────
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "jax[tpu]",
    "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html"
], check=True)

subprocess.run([
    "pip", "install", "-q",
    "flax>=0.8.2",
    "optax>=0.2.2",
    "orbax-checkpoint>=0.5.10",
    "transformers>=4.40.0",
    "sentencepiece>=0.2.0",
    "librosa>=0.10.1",
    "soundfile>=0.12.1",
    "whisper-timestamped>=1.15.4",
], check=True)

print("Dependencies installed.")


# ── CELL 2: TPU setup ────────────────────────────────────────────
import sys

sys.path.insert(0, "/kaggle/input/urdu-moshi-code/urdu_moshi")
from kaggle_utils.tpu_setup import setup_kaggle_tpu, get_kaggle_paths, print_kaggle_quota_reminder

print_kaggle_quota_reminder()
num_devices = setup_kaggle_tpu()


# ── CELL 3: Paths ────────────────────────────────────────────────
paths = get_kaggle_paths(base_dataset_name=None)
paths["audio_dir"] = "/kaggle/working/audio"
paths["pretrained_mimi"] = "/kaggle/working/mimi_weights"
paths["qwen_weights"] = "/kaggle/working/qwen25_weights"
print("Paths:")
for k, v in paths.items():
    print(f"  {k}: {v}")


# ── CELL 4: Config ───────────────────────────────────────────────
from kaggle_utils.kaggle_configs import KaggleStage1Config
from configs.model_config import MoshiConfig

train_cfg = KaggleStage1Config(
    data_path=paths["audio_dir"],
    checkpoint_dir=paths["checkpoint_dir"],
)
model_cfg = MoshiConfig()
print(f"Training for {train_cfg.total_steps} steps")
print(f"Saving every {train_cfg.save_every_steps} steps")


# ── CELL 5: Load Urdu tokenizer ──────────────────────────────────
from tokenizer.urdu_tokenizer import UrduTokenizer

TOKENIZER_PATH = "/kaggle/input/urdu-moshi-tokenizer/urdu_spm_32k.model"
tokenizer = UrduTokenizer(TOKENIZER_PATH)
print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")


# ── CELL 6: Load backbone weights ────────────────────────────────
from model.backbone import load_qwen25_pretrained_weights
from configs.model_config import Qwen25BackboneConfig

backbone_cfg = Qwen25BackboneConfig()
backbone_pretrained = load_qwen25_pretrained_weights(
    config=backbone_cfg,
    cache_dir=paths["qwen_weights"],
)
print("Backbone weights loaded.")


# ── CELL 7: Load Mimi codec ──────────────────────────────────────
from mimi.codec import create_mimi_for_finetuning
from configs.model_config import MimiConfig
import jax
import jax.numpy as jnp

mimi_cfg = MimiConfig()
mimi_model, mimi_pretrained = create_mimi_for_finetuning(
    config=mimi_cfg,
    pretrained_checkpoint=paths["pretrained_mimi"],
    freeze_encoder=False,
    freeze_decoder=False,
)

MIMI_PARAMS = mimi_pretrained

def mimi_encode_fn(audio_jnp):
    result = mimi_model.apply(
        {"params": MIMI_PARAMS},
        audio_jnp,
        train=False,
    )
    return result["tokens"]

print("Mimi codec ready.")


# ── CELL 8: Resume from previous session if available ────────────
from kaggle_utils.kaggle_checkpoint import KaggleCheckpointManager

PREVIOUS_SESSION_DATASET = "/kaggle/input/urdu-moshi-stage1"

ckpt_manager = KaggleCheckpointManager(
    checkpoint_dir=train_cfg.checkpoint_dir,
    max_to_keep=2,
)

resume_step = None
if __import__("os").path.exists(PREVIOUS_SESSION_DATASET):
    resume_step = KaggleCheckpointManager.load_from_previous_session(
        previous_dataset_path=f"{PREVIOUS_SESSION_DATASET}/checkpoints/stage1",
        working_dir=train_cfg.checkpoint_dir,
    )
    print(f"Will resume from step {resume_step}")
else:
    print("Starting fresh (no previous session found).")


# ── CELL 9: Build dataset ────────────────────────────────────────
from data.audio_pipeline import Stage1Dataset
from configs.model_config import MultiStreamConfig, InnerMonologueConfig

WHISPER_MODEL = None

def whisper_transcribe_fn(file_path: str, speaker=None):
    if WHISPER_MODEL is None:
        return None
    return WHISPER_MODEL.transcribe(file_path, word_timestamps=True)

dataset = Stage1Dataset(
    audio_dir=train_cfg.data_path,
    mimi_encode_fn=mimi_encode_fn,
    tokenizer=tokenizer,
    whisper_transcribe_fn=whisper_transcribe_fn,
    ms_config=MultiStreamConfig(),
    im_config=InnerMonologueConfig(
        text_vocab_size=tokenizer.vocab_size,
    ),
    segment_duration=train_cfg.sequence_duration_seconds,
    text_mask_prob=train_cfg.text_mask_prob,
    text_delay_min=train_cfg.text_delay_min,
    text_delay_max=train_cfg.text_delay_max,
    acoustic_delay=train_cfg.acoustic_delay,
    batch_size=num_devices,
)
print("Dataset pipeline ready.")


# ── CELL 10: Build and train the model ───────────────────────────
from model.rq_transformer import RQTransformer
from training.trainer import MoshiTrainer

model = RQTransformer(config=model_cfg.rq_transformer)

pretrained_params = {"backbone": backbone_pretrained}

trainer = MoshiTrainer(
    model=model,
    config=train_cfg,
    stage=1,
    pretrained_params=pretrained_params,
    checkpoint_dir=train_cfg.checkpoint_dir,
)

if resume_step is not None:
    restored = ckpt_manager.restore(step=resume_step)
    if restored:
        trainer.state = jax.device_put_replicated(restored, jax.devices())
        print(f"Resumed training state from step {resume_step}")

trainer.checkpoint_manager = ckpt_manager

trainer.train(data_iterator=dataset.iterate())


# ── CELL 11: Save completion status ──────────────────────────────
from kaggle_utils.kaggle_checkpoint import KaggleCheckpointManager
KaggleCheckpointManager.print_commit_instructions(stage=1)

print("""
NEXT STEPS:
  1. Save this notebook's working directory as a Kaggle Dataset
     named 'urdu-moshi-stage1'
  2. For Stage 2, create a new notebook:
     - Add 'urdu-moshi-stage1' as input dataset
     - Set init_from_checkpoint in KaggleStage2Config
     - Run stage2 notebook
""")
