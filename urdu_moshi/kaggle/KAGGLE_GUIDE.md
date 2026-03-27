# Kaggle TPU v5e-8 Training Guide

## Key Facts About Kaggle TPU

| | Details |
|---|---|
| **TPU type** | v5e-8 (Kaggle phased out v3-8 — your code targets the right hardware) |
| **Chips** | 8 chips × 16GB HBM = 128GB total |
| **TPU quota** | ~20 hours / week (free tier) |
| **Session cap** | 12 hours max per notebook run |
| **Working disk** | 20GB at `/kaggle/working` |
| **Internet** | Available during session |
| **VS Code** | Write code locally, execute on Kaggle web UI — no direct connection possible |

---

## VS Code Workflow

```
Local Machine (VS Code)              Kaggle Cloud (TPU v5e-8)
──────────────────────────           ──────────────────────────
Write / edit code                    Execute notebooks
Run syntax checks                    Access TPU hardware
Use git for version control          Save to /kaggle/working
                │                              │
                │  Upload via Kaggle API       │
                └──────────────────────────────►
                                     kaggle datasets create
                                          or
                                     copy-paste into notebook
```

You **cannot** remotely attach VS Code to a Kaggle kernel.
The Kaggle web notebook is the only execution environment.

---

## One-Time Setup (Do This Before First Training Run)

### 1. Install Kaggle API locally
```bash
pip install kaggle
# Add your kaggle.json API key to ~/.kaggle/kaggle.json
```

### 2. Upload your project code as a Kaggle Dataset
```bash
cd /path/to/urdu_moshi/
kaggle datasets create \
  --dir-mode zip \
  -p . \
  --title "urdu-moshi-code" \
  --slug urdu-moshi-code
```

After the first upload, push updates with:
```bash
kaggle datasets version -p . -m "update description"
```

### 3. Upload your Urdu speech data as a Kaggle Dataset
```bash
# Structure your data like this before uploading:
urdu-speech-corpus/
├── audio/              ← .wav or .flac files, mono, 24kHz
│   ├── speaker1/
│   └── speaker2/
├── transcripts/        ← Whisper transcript .json files
├── mimi_checkpoint/    ← pretrained Mimi weights (download separately)
└── qwen25_weights/     ← Qwen2.5-1.5B weights (download separately)

kaggle datasets create --dir-mode zip -p ./urdu-speech-corpus
```

### 4. Upload pre-downloaded model weights
Kaggle has internet but downloads are slow and count against session time.
Pre-download and upload as a dataset:

```bash
# Download Qwen2.5-1.5B locally
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B')
model.save_pretrained('./qwen25_weights')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B').save_pretrained('./qwen25_weights')
"
kaggle datasets create --dir-mode zip -p ./qwen25_weights --slug qwen25-weights
```

---

## Notebook Setup (Do This Each Training Session)

1. Go to [kaggle.com/code](https://kaggle.com/code) → **New Notebook**
2. Right sidebar → **Session options**:
   - Accelerator: **TPU VM v5e-8**
   - Internet: **On**
3. **Add data** (top right → "Add data"):
   - `urdu-moshi-code` (your project code)
   - `urdu-speech-corpus` (your audio data)
   - `qwen25-weights` (pretrained weights)
   - `urdu-moshi-tokenizer` (your trained SentencePiece model)
   - `urdu-moshi-stage1` ← add this when resuming from a previous session
4. Paste the contents of `kaggle/stage1_notebook.py` into the notebook cells
5. **Run all**

---

## Inter-Session Workflow (Critical — Read This)

Each Kaggle session is isolated. After a session ends, `/kaggle/working` is
**deleted**. You must commit checkpoints to a Kaggle Dataset before the session ends.

```
Session 1 (12h)
├── Train Stage 1 for 50,000 steps
├── Checkpoints saved to /kaggle/working/checkpoints/stage1/
└── BEFORE SESSION ENDS:
    Notebook → "Save & Run" → "Save working directory as Dataset"
    Name: urdu-moshi-stage1
    ─────────────────────────────────────────────────
Session 2 (12h)
├── Add dataset "urdu-moshi-stage1" as input
├── load_from_previous_session() copies ckpt to /kaggle/working
├── trainer resumes from last saved step
└── Continue training or move to Stage 2
```

**Set a 30-minute reminder** before your session ends to commit the dataset.
Kaggle shows session time remaining in the top right.

---

## File Paths Reference

| What | Path |
|---|---|
| Your code | `/kaggle/input/urdu-moshi-code/urdu_moshi/` |
| Urdu audio | `/kaggle/input/urdu-speech-corpus/audio/` |
| Pretrained Mimi | `/kaggle/input/urdu-speech-corpus/mimi_checkpoint/` |
| Qwen2.5 weights | `/kaggle/input/qwen25-weights/` |
| Urdu tokenizer | `/kaggle/input/urdu-moshi-tokenizer/urdu_spm_32k.model` |
| Stage 1 resume | `/kaggle/input/urdu-moshi-stage1/checkpoints/stage1/` |
| Stage 2 resume | `/kaggle/input/urdu-moshi-stage2/checkpoints/stage2/` |
| Your checkpoints | `/kaggle/working/checkpoints/` |
| TensorBoard logs | `/kaggle/working/tensorboard/` |

---

## What Changed From the Original Code

The architecture files are **100% unchanged**. Only these 3 things are Kaggle-specific:

| File | What it does |
|---|---|
| `kaggle/tpu_setup.py` | Calls `jax.distributed.initialize()` correctly for Kaggle's environment |
| `kaggle/kaggle_configs.py` | Overrides paths and reduces batch/step counts for 12h sessions |
| `kaggle/kaggle_checkpoint.py` | Handles 20GB disk limit, auto-prunes, cross-session resume |
| `kaggle/stage1_notebook.py` | Ready-to-paste notebook cells for Stage 1 |

When you move to Google Cloud TPU v5e-8 in the future, just use the original
`configs/train_configs/stage1_pretrain.py` with GCS paths — nothing else changes.

---

## Realistic Training Timeline on Kaggle (Free Tier)

| Stage | Steps | Hours | Sessions Needed |
|---|---|---|---|
| Tokenizer training | — | 1h (CPU) | 1 |
| Mimi fine-tune | 50k | ~8h | 1 |
| Stage 1 (audio pretrain) | 50k | ~10h | 1–2 |
| Stage 2 (multi-stream) | 20k | ~4h | 1 |
| Stage 3 (conversation) | 5k | ~1h | 1 |
| Stage 4 (instruct) | 10k | ~2h | 1 |
| **Total** | | **~26h** | **~4–6 sessions** |

At 20h/week quota, you can complete the full pipeline in **2 weeks**.
This is a scaled-down version — the paper used 1M steps for Stage 1.
For research-quality results, you will eventually need Google Cloud TPU.
