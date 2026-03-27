# Urdu Moshi — Full-Duplex Urdu Speech Dialogue System

A faithful JAX/Flax re-implementation of the **Moshi** architecture (Défossez et al., 2024)
designed from the ground up for **Urdu language** and **TPU v5e-8** training.

The middle backbone (Temporal Transformer) is a **hot-swappable abstract interface** —
plug in any Urdu LLM you build later without touching any other component.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         URDU MOSHI PIPELINE                          │
│                                                                       │
│  AUDIO IN (24kHz)                         AUDIO OUT (24kHz)          │
│       │                                         ▲                    │
│       ▼                                         │                    │
│  ┌─────────┐   tokens (S × 8)   ┌────────────────────────────────┐  │
│  │  MIMI   │──────────────────▶ │     RQ-TRANSFORMER CORE         │  │
│  │ Encoder │                    │                                  │  │
│  │(causal) │   user stream      │  ┌──────────────────────────┐   │  │
│  └─────────┘──────────────────▶ │  │  TEMPORAL TRANSFORMER    │   │  │
│                                  │  │  (Qwen2.5-1.5B Flax)     │   │  │
│                                  │  │                           │   │  │
│  URDU TEXT                       │  │  ◀── YOUR LLM GOES HERE   │   │  │
│  TOKENIZER                       │  │  AbstractTemporalBackbone │   │  │
│  (SentencePiece                  │  └────────────┬─────────────┘   │  │
│   32k vocab)                     │               │ context vectors  │  │
│       │                          │               ▼                  │  │
│       │ Inner                    │  ┌──────────────────────────┐   │  │
│       │ Monologue                │  │   DEPTH TRANSFORMER      │   │  │
│       │ text tokens              │  │   (6 layers, 1024 dim)   │   │  │
│       └─────────────────────────▶│  │   K=17 depthwise heads   │   │  │
│                                  │  │   per-codebook params    │   │  │
│                                  │  └────────────┬─────────────┘   │  │
│                                  │               │ 17 logit streams │  │
│                                  └───────────────┼────────────────┘  │
│                                                  │                    │
│                                    ┌─────────────▼──────────────┐    │
│                                    │   JOINT TOKEN SEQUENCE      │    │
│                                    │                             │    │
│                                    │  k=0   : Urdu text token    │    │
│                                    │  k=1   : Moshi semantic     │    │
│                                    │  k=2-8 : Moshi acoustic×7   │    │
│                                    │  k=9   : User semantic      │    │
│                                    │  k=10-16: User acoustic×7   │    │
│                                    └─────────────┬───────────────┘    │
│                                                  │                    │
│                                                  ▼                    │
│                                          ┌───────────┐               │
│                                          │   MIMI    │               │
│                                          │  Decoder  │               │
│                                          │  (causal) │               │
│                                          └───────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions (paper-faithful)

| Component | Paper | This Implementation |
|---|---|---|
| Frame rate | 12.5 Hz | 12.5 Hz ✓ |
| Codec | Mimi (pretrained) | Mimi fine-tuned on Urdu ✓ |
| RVQ | 1 semantic VQ + 7 acoustic | Split RVQ identical ✓ |
| Acoustic delay | τ=2 (pretrain), τ=1 (finetune) | Configurable per-stage ✓ |
| Text alignment | Whisper word-level timestamps | Whisper Urdu transcription ✓ |
| Depth Transformer | 6 layers, depthwise params | Identical ✓ |
| Semantic loss weight | ×100 vs acoustic | ×100 ✓ |
| Training stages | 4 stages | All 4 stages ✓ |
| Backbone | Helium 7B (not released) | **Qwen2.5-1.5B (swappable)** ✓ |
| Target hardware | H100 GPU | **TPU v5e-8** ✓ |

---

## Repository Structure

```
urdu_moshi/
│
├── configs/
│   ├── model_config.py           ← All architecture hyperparams (dataclasses)
│   └── train_configs/
│       ├── stage1_pretrain.py    ← Single-stream audio pretraining
│       └── stage2_3_4.py         ← Multi-stream, Fisher, Instruct finetuning
│
├── tokenizer/
│   └── urdu_tokenizer.py         ← SentencePiece Urdu tokenizer + text alignment
│
├── mimi/
│   ├── encoder.py                ← Causal SeaNet encoder + Transformer bottleneck
│   ├── decoder.py                ← Causal transposed-conv decoder + Transformer
│   ├── quantizer.py              ← Split RVQ: 1 semantic VQ + 7 acoustic VQs
│   └── codec.py                  ← Full Mimi model + pretrained weight loading
│
├── model/
│   ├── backbone.py               ← AbstractTemporalBackbone + Qwen2.5 Flax impl
│   ├── depth_transformer.py      ← 6-layer Depth Transformer, depthwise per-codebook
│   ├── rq_transformer.py         ← Full RQ-Transformer + acoustic delay + joint seq
│   └── inner_monologue.py        ← Text-audio frame alignment + multi-stream builder
│
├── training/
│   ├── loss.py                   ← Weighted CE loss (semantic×100, text, acoustic)
│   ├── scheduler.py              ← LR schedules for all 4 stages
│   └── trainer.py                ← TPU v5e-8 pmap training loop + EMA
│
├── data/
│   └── audio_pipeline.py         ← Urdu audio datasets for all 4 stages
│
├── checkpointing/
│   └── checkpoint_manager.py     ← orbax checkpointing + backbone hot-swap API
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

```bash
# 1. Create environment
conda create -n urdu_moshi python=3.12 -y
conda activate urdu_moshi

# 2. Install JAX with TPU support
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Install this package
pip install -e .
```

---

## Quickstart: Train on TPU v5e-8

### Step 0 — Train the Urdu Tokenizer

```python
from tokenizer.urdu_tokenizer import build_urdu_tokenizer_from_whisper_transcripts

tokenizer = build_urdu_tokenizer_from_whisper_transcripts(
    transcript_dir="gs://your-bucket/urdu_transcripts/",
    output_prefix="urdu_spm_32k",
    vocab_size=32000,
)
```

> **Data requirement:** At minimum 50M Urdu tokens of text.
> Sources: OPUS Urdu corpora, UrduHack dataset, BBC Urdu, Dawn News crawl.

---

### Step 1 — Fine-tune Mimi on Urdu Speech

```python
from mimi.codec import create_mimi_for_finetuning
from configs.model_config import MimiConfig

mimi, pretrained_params = create_mimi_for_finetuning(
    config=MimiConfig(),
    pretrained_checkpoint="kyutai/moshika-pytorch-bf16",
    freeze_encoder=False,
    freeze_decoder=False,
)
```

> **Why fine-tune Mimi?** While Mimi is language-agnostic acoustically,
> Urdu has unique phoneme distributions (retroflex consonants, aspirated stops,
> nasalized vowels). Fine-tuning on 500–1000h of Urdu speech improves
> the semantic VQ's phonetic discriminability (measured by ABX error rate).

---

### Step 2 — Load Qwen2.5 Backbone Weights

```python
from model.backbone import Qwen25FlaxBackbone, load_qwen25_pretrained_weights
from configs.model_config import Qwen25BackboneConfig

backbone_config = Qwen25BackboneConfig(
    hf_model_id="Qwen/Qwen2.5-1.5B",
)

pretrained_backbone_params = load_qwen25_pretrained_weights(
    config=backbone_config,
    cache_dir="/tmp/hf_cache",
)
```

---

### Step 3 — Stage 1: Single-Stream Audio Pretraining

```python
from configs.train_configs.stage1_pretrain import Stage1TrainConfig
from configs.model_config import MoshiConfig, RQTransformerConfig
from model.rq_transformer import RQTransformer
from training.trainer import run_training_stage
from data.audio_pipeline import Stage1Dataset

cfg_train = Stage1TrainConfig(
    data_path="gs://your-bucket/urdu_audio/stage1",
    checkpoint_dir="gs://your-bucket/checkpoints/stage1",
)

model_cfg = MoshiConfig()
model = RQTransformer(config=model_cfg.rq_transformer)

dataset = Stage1Dataset(
    audio_dir=cfg_train.data_path,
    mimi_encode_fn=mimi_encode_fn,
    tokenizer=tokenizer,
    whisper_transcribe_fn=whisper_fn,
    ms_config=model_cfg.rq_transformer.multistream,
    im_config=model_cfg.rq_transformer.inner_monologue,
    acoustic_delay=cfg_train.acoustic_delay,
    batch_size=8 * 8,
)

final_state = run_training_stage(
    stage=1,
    model=model,
    config=cfg_train,
    data_iterator=dataset.iterate(),
    pretrained_params=pretrained_backbone_params,
    checkpoint_dir=cfg_train.checkpoint_dir,
)
```

---

### Step 4 — Stages 2 → 3 → 4 (Progressive Fine-tuning)

```python
from configs.train_configs.stage2_3_4 import (
    Stage2TrainConfig, Stage3TrainConfig, Stage4TrainConfig
)
from data.audio_pipeline import Stage2Dataset

for Stage, DatasetClass, stage_num in [
    (Stage2TrainConfig, Stage2Dataset, 2),
    (Stage3TrainConfig, Stage2Dataset, 3),
    (Stage4TrainConfig, Stage2Dataset, 4),
]:
    cfg = Stage()
    dataset = DatasetClass(...)

    final_state = run_training_stage(
        stage=stage_num,
        model=model,
        config=cfg,
        data_iterator=dataset.iterate(),
        pretrained_params=None,
        checkpoint_dir=cfg.checkpoint_dir,
    )
```

---

## Swapping in Your Own Backbone (Key Feature)

When your custom Urdu LLM is ready, swapping it in requires **zero changes** to
the RQ-Transformer, Depth Transformer, Mimi, or training code.

### Contract your backbone must satisfy

```python
from model.backbone import AbstractTemporalBackbone
import jax.numpy as jnp
from typing import Tuple

class YourUrduLLM(AbstractTemporalBackbone):

    @property
    def model_dim(self) -> int:
        return 2048  # your hidden size

    @property
    def vocab_size(self) -> int:
        return 32000  # must match Urdu tokenizer

    @property
    def max_seq_len(self) -> int:
        return 4096

    def __call__(
        self,
        embedded_tokens: jnp.ndarray,  # (B, S, embed_dim)
        train: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Returns:
        #   context_vectors: (B, S, model_dim)   ← fed to Depth Transformer
        #   text_logits:     (B, S, vocab_size)  ← Inner Monologue text prediction
        ...
```

### Hot-swap at any training stage

```python
from checkpointing.checkpoint_manager import CheckpointManager

ckpt = CheckpointManager("gs://your-bucket/checkpoints/stage2")
old_state = ckpt.restore(step=100_000)

new_backbone = YourUrduLLM(...)
new_backbone_params = ...  # load your pretrained weights

import optax
new_opt = optax.adamw(learning_rate=2e-6)
new_opt_state = new_opt.init(new_backbone_params)

new_state = ckpt.swap_backbone(
    current_state=old_state,
    new_backbone_params=new_backbone_params,
    new_temporal_optimizer_state=new_opt_state,
)
```

> ✅ Step counter preserved
> ✅ Depth Transformer params preserved
> ✅ EMA weights updated
> ✅ Just update model_config.py → point `Qwen25BackboneConfig` to your class

---

## Training Data Requirements

### Stage 1 — Single Stream Audio Pretrain
- **Volume:** ~7M hours (paper). Practical minimum: **500h** for meaningful convergence.
- **Format:** Any Urdu speech, mono, resampled to 24kHz.
- **Transcription:** Whisper `large-v3` with `whisper-timestamped` for word-level alignment.
- **Urdu sources:** Common Voice Urdu, VoxPopuli, OpenSLR Urdu, scraped podcast audio.

### Stage 2 — Multi-Stream (Diarized)
- **Volume:** ~7M hours with diarization. Practical: **200h** diarized Urdu conversations.
- **Diarization:** pyannote-audio 3.x (`pyannote/speaker-diarization-3.1`).

### Stage 3 — Conversation Fine-tuning (Fisher-equivalent)
- **Volume:** 2000h two-channel phone conversations (paper). For Urdu: **50–100h**.
- **Format:** Two-channel audio with separate speaker tracks.
- **Urdu sources:** Record or source two-speaker Urdu conversations with separate mics.

### Stage 4 — Instruction Fine-tuning
- **Volume:** 20k+ hours synthetic (paper). For Urdu: **1000–5000h synthetic**.
- **Method:** Generate Urdu dialogue transcripts with your Urdu LLM,
  synthesize with a streaming Urdu TTS engine, use as training data.

---

## TPU v5e-8 Configuration Notes

The TPU v5e-8 gives 8 chips × ~16GB HBM = **~128GB total HBM**.

| Component | Memory footprint (bf16) | Notes |
|---|---|---|
| Qwen2.5-1.5B params | ~3GB | Fits on 1 chip |
| Depth Transformer | ~0.5GB | Tiny, fits anywhere |
| Activations (S=3000, B=1) | ~12GB | Per chip with gradient checkpointing |
| Optimizer states | ~6GB | AdamW 2× params |
| **Total per chip** | **~21GB** | Within 16GB with gradient checkpointing |

**Gradient checkpointing is enabled by default** in `Qwen25BackboneConfig`.
This trades ~40% compute for ~60% memory savings on activations.

To adjust batch size for your HBM budget:

```python
# In stage1_pretrain.py
# batch_audio_hours=16 → ~8 sequences of 5min each → batch_size=8 total
# With 8 TPU chips: 1 sequence per chip
cfg.batch_size = 8  # 1 per chip × 8 chips
```

---

## Citation

If you use this codebase, please cite the original Moshi paper:

```bibtex
@techreport{kyutai2024moshi,
    title={Moshi: a speech-text foundation model for real-time dialogue},
    author={Alexandre Défossez and Laurent Mazaré and Manu Orsini and
    Amélie Royer and Patrick Pérez and Hervé Jégou and Edouard Grave and Neil Zeghidour},
    year={2024},
    eprint={2410.00037},
    archivePrefix={arXiv},
}
```
