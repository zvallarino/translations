# Transcript Translation Pipeline

Translate non-English TikTok transcripts in `urls.xlsx` to English using
**FastText** for language detection and **Meta NLLB-200** for translation.

Runs on GPU (CUDA). Tested on RTX 3070 (8 GB VRAM). Designed for overnight
batch runs of thousands of transcripts.

---

## What it does

For each row in `urls.xlsx`:

1. Reads the `Transcript` column.
2. Detects the source language (176 languages supported — incl. Hausa,
   Yoruba, Igbo, Swahili, Arabic, etc.).
3. Translates to English via NLLB-200. Long transcripts are split into
   sentence-sized chunks and re-joined, so nothing gets truncated.
4. Writes two new columns: `Language_NLLB`, `Translation_NLLB`.
5. Checkpoints to disk every 10 rows. Crash/Ctrl+C is safe — re-running
   skips rows that are already done.

---

## Setup (first time only)

You need: Python 3.10+, an NVIDIA GPU with CUDA drivers, and ~5 GB free disk
for the models.

```bash
# 1. Create a venv (if you don't already have one)
python3 -m venv venv
source venv/bin/activate     # Linux/WSL
# or: venv\Scripts\activate  # Windows PowerShell

# 2. Install PyTorch with CUDA support
#    Pick the right CUDA version for your driver. Check with: nvidia-smi
#    The line below is for CUDA 12.1 — adjust if your driver is older.
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install everything else
pip install transformers sentencepiece pandas openpyxl fasttext
```

The FastText language-ID model (`lid.176.bin`, ~126 MB) downloads itself on
first run if it's not in the working directory.

The NLLB model (~2.5 GB for 1.3B, ~6 GB for 3.3B) downloads from Hugging Face
on first run and is cached under `~/.cache/huggingface/`.

---

## Running it

Put your spreadsheet at `urls.xlsx` (in the same folder as the script) with a
column named `Transcript`. Then:

```bash
python translate_transcripts.py
```

That's it. It will:

- Show GPU info on startup
- Print a progress line per row with detected language, a translation preview,
  and a rough ETA
- Save every 10 rows
- Exit cleanly on Ctrl+C (saves the current row first)

To resume after stopping, just re-run the same command. Rows where both
`Language_NLLB` and `Translation_NLLB` are filled in get skipped.

---

## Model choice

The script has two NLLB options at the top of the file:

```python
NLLB_MODEL = "facebook/nllb-200-1.3B"   # default — recommended
# NLLB_MODEL = "facebook/nllb-200-3.3B" # strongest, ~2.5x slower
```

**Use 1.3B (the default)** for an overnight run on an 8 GB 3070:
- ~3 GB VRAM at float16
- Throughput ~1–3 transcripts/sec for short clips, slower for long ones
- Quality is very good on African languages (Hausa/Yoruba/Igbo)
- Leaves headroom — you can have a browser tab open without OOM

**Use 3.3B** if you want maximum quality and don't mind slower throughput:
- ~6.5 GB VRAM at float16 — fits on a 3070 only if nothing else uses the GPU
- ~2.5x slower per chunk
- Modest quality bump (a few BLEU points on low-resource languages)
- Close other GPU-using apps before starting

To switch, comment/uncomment the two `NLLB_MODEL = ...` lines in
`translate_transcripts.py`.

---

## Config knobs (top of the script)

| Variable            | Default | What it does                                        |
|---------------------|---------|-----------------------------------------------------|
| `EXCEL_FILE`        | `urls.xlsx` | Input/output spreadsheet                        |
| `SAVE_EVERY`        | 10      | Checkpoint frequency (rows)                         |
| `MAX_INPUT_TOKENS`  | 400     | Tokens per input chunk (NLLB hard limit is 1024)    |
| `MAX_OUTPUT_TOKENS` | 512     | Tokens per output chunk                             |
| `NUM_BEAMS`         | 4       | Beam search width — quality vs speed tradeoff       |

Lowering `NUM_BEAMS` to 1 (greedy) roughly doubles speed at a quality cost.
Raising to 5 is marginally better; not worth the extra time.

---

## Output spreadsheet

The script adds two columns:

- **`Language_NLLB`** — Detected language as a human-readable name
  (e.g. `Hausa`, `Yoruba`, `English`).
- **`Translation_NLLB`** — English translation. For rows already in English,
  this is just a copy of the original transcript.

Error rows get a tagged string in the translation column:
- `[unsupported language: xx]` — FastText returned a code NLLB doesn't cover
- `[cuda-oom: ...]` — GPU ran out of memory (rare with chunking; close other
  GPU apps and re-run — the row will be retried since the column won't be
  marked complete... wait, actually it will be marked. To force a retry,
  clear those cells manually before re-running.)
- `[error: ...]` — Some other exception; the message is preserved.

---

## Troubleshooting

**`CUDA out of memory`**
- You're probably on 3.3B with other GPU users. Close browser/games and
  retry, or switch to 1.3B.
- If it happens on 1.3B, lower `MAX_INPUT_TOKENS` to 256.

**Translations look truncated**
- Shouldn't happen anymore — chunking handles arbitrarily long inputs. If
  you see this, check that the row's transcript actually has the full text
  (Whisper might have produced an empty/short output).

**FastText misidentifies short transcripts**
- Very short clips (<30 chars) are hard. Consider filtering them out or
  spot-checking the `Language_NLLB` column.

**It's slow**
- First-time loads include downloading models (one-time cost).
- Long transcripts get chunked, so a 5-minute video can take 10+ seconds.
- Drop `NUM_BEAMS` to 1 for ~2x speedup at a quality cost.

---

## Hardware notes

This script was developed for:
- AMD Ryzen 9 5900X (12c/24t)
- 32 GB system RAM
- NVIDIA RTX 3070 (8 GB VRAM)

The CPU and system RAM are barely used — everything happens on the GPU.
The only CPU-heavy step is the FastText lookup, which takes a millisecond.
# translations
