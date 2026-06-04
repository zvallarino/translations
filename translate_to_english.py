#!/usr/bin/env python3
"""
Step 2 of 2 — Translate non-English transcripts to English.
 
Reads `urls.xlsx` (which must already have Language_NLLB_Code populated by
detect_language.py first), and for every row whose language is NOT English,
runs Meta's NLLB-200 on GPU to produce an English translation.
 
For English rows: copies the transcript through unchanged.
For rows with no detected language or empty transcript: skipped.
 
Writes to a new column `Translation_NLLB`.
 
Long transcripts get split into sentence-sized chunks before translation
since NLLB has a 1024-token ceiling. Chunks are batched through the model,
sorted by length to minimize padding waste, then joined back in order.
 
Saves every SAVE_EVERY rows. Ctrl-C exits safely. Re-running skips rows
already translated.
"""
 
import os
import re
import signal
import sys
import time
import warnings
 
warnings.filterwarnings("ignore")
 
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
 
from lang_codes import FT_TO_NLLB
 
# ── Config ────────────────────────────────────────────────────────────────
EXCEL_FILE = "urls.xlsx"
 
# Pick ONE model. 1.3B is the recommended default on an 8 GB 3070.
# 3.3B is stronger but needs ~6.5 GB VRAM at float16 and is ~2.5x slower.
NLLB_MODEL = "facebook/nllb-200-1.3B"
# NLLB_MODEL = "facebook/nllb-200-3.3B"
 
TRANSCRIPT_COL = "Transcript"
LANG_CODE_COL  = "Language_NLLB_Code"
TRANS_COL      = "Translation_NLLB"
 
SAVE_EVERY        = 50    # Excel writes on 13k rows aren't free; bigger = faster
MAX_INPUT_TOKENS  = 400   # per chunk into the model (NLLB hard limit is 1024)
MAX_OUTPUT_TOKENS = 512   # per chunk out
NUM_BEAMS         = 1     # beam=4 barely helps on noisy YouTube ASR; beam=1 is ~2-3x faster
BATCH_SIZE        = 8     # chunks per GPU batch; drop to 4 if CUDA OOM
 
 
# ── Graceful Ctrl-C ───────────────────────────────────────────────────────
_stop = False
def _on_sigint(sig, frame):
    global _stop
    print("\n[!] Ctrl-C — saving after this row then exiting.", flush=True)
    _stop = True
signal.signal(signal.SIGINT, _on_sigint)
 
 
# Sentence-ish splitter: ends on . ! ? newline; also handles … and 。
_SENT_SPLIT = re.compile(r"(?<=[\.!\?…。])\s+|\n+")
 
 
def split_into_chunks(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Split `text` into chunks where each chunk tokenizes to <= max_tokens.
 
    First splits on sentence boundaries, then greedily packs sentences into
    chunks. Sentences longer than max_tokens are further word-split as a
    fallback. Good enough for messy transcripts; a full sentence tokenizer
    is overkill here.
    """
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        return []
 
    def tok_len(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=False).input_ids)
 
    chunks = []
    current, current_len = [], 0
 
    for sent in sentences:
        sent_len = tok_len(sent)
 
        if sent_len > max_tokens:
            # Single sentence too long → fall back to word-splitting.
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            buf, buf_len = [], 0
            for w in sent.split():
                wlen = tok_len(w + " ")
                if buf_len + wlen > max_tokens and buf:
                    chunks.append(" ".join(buf))
                    buf, buf_len = [], 0
                buf.append(w)
                buf_len += wlen
            if buf:
                chunks.append(" ".join(buf))
            continue
 
        if current_len + sent_len > max_tokens and current:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += sent_len
 
    if current:
        chunks.append(" ".join(current))
    return chunks
 
 
def translate(tokenizer, model, device, text: str, src_code: str) -> str:
    nllb_src = FT_TO_NLLB.get(src_code)
    if not nllb_src:
        return f"[unsupported language: {src_code}]"
 
    try:
        tokenizer.src_lang = nllb_src
        chunks = split_into_chunks(text, tokenizer, MAX_INPUT_TOKENS)
        if not chunks:
            return ""
 
        # Sort chunks by length so each batch contains similar-length items.
        # This minimizes padding waste. We remember the original order so we
        # can reassemble the translated text in the right sequence.
        indexed = sorted(enumerate(chunks), key=lambda x: len(x[1]))
        order = [i for i, _ in indexed]
        sorted_chunks = [c for _, c in indexed]
 
        tgt_id = tokenizer.convert_tokens_to_ids("eng_Latn")
        translated_sorted = [None] * len(sorted_chunks)
 
        for i in range(0, len(sorted_chunks), BATCH_SIZE):
            batch = sorted_chunks[i : i + BATCH_SIZE]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_TOKENS + 32,
                padding=True,
            ).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_id,
                    max_length=MAX_OUTPUT_TOKENS,
                    num_beams=NUM_BEAMS,
                    no_repeat_ngram_size=3,
                )
            decoded = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            for j, out in enumerate(decoded):
                translated_sorted[i + j] = out
 
        # Reassemble in original chunk order.
        outputs = [None] * len(chunks)
        for sorted_idx, original_idx in enumerate(order):
            outputs[original_idx] = translated_sorted[sorted_idx]
 
        return " ".join(outputs).strip()
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return f"[cuda-oom: {exc}]"
    except Exception as exc:
        return f"[error: {exc}]"
 
 
def main():
    if not os.path.exists(EXCEL_FILE):
        sys.exit(f"[error] {EXCEL_FILE} not found in current directory.")
 
    print(f"Reading {EXCEL_FILE} …", flush=True)
    df = pd.read_excel(EXCEL_FILE)
    df.columns = df.columns.str.strip()
 
    # Verify the prerequisite column exists.
    if LANG_CODE_COL not in df.columns:
        sys.exit(
            f"[error] No '{LANG_CODE_COL}' column in {EXCEL_FILE}.\n"
            f"        Run detect_language.py first."
        )
    if TRANSCRIPT_COL not in df.columns:
        sys.exit(f"[error] No '{TRANSCRIPT_COL}' column in {EXCEL_FILE}.")
 
    if TRANS_COL not in df.columns:
        df[TRANS_COL] = pd.NA
    df[TRANS_COL] = df[TRANS_COL].astype(object)
 
    # Figure out what we actually need to do BEFORE loading the model. This
    # lets us skip GPU setup entirely if there's nothing left to translate
    # (e.g. on resume after everything finished).
    needs_translation = []  # list of (idx, lang_code, text) tuples
    english_pending   = []  # English rows that still need transcript copied across
    skipped_empty     = 0
    skipped_no_lang   = 0
    already_done      = 0
 
    for idx in range(len(df)):
        transcript = df.at[idx, TRANSCRIPT_COL]
        if pd.isna(transcript) or str(transcript).strip() == "":
            skipped_empty += 1
            continue
 
        if pd.notna(df.at[idx, TRANS_COL]):
            already_done += 1
            continue
 
        lang_code = df.at[idx, LANG_CODE_COL]
        if pd.isna(lang_code) or not str(lang_code).strip():
            skipped_no_lang += 1
            continue
 
        lang_code = str(lang_code).strip().lower()
        text = str(transcript).strip()
 
        if lang_code == "en":
            english_pending.append((idx, text))
        else:
            needs_translation.append((idx, lang_code, text))
 
    print(
        f"  {already_done} rows already translated (skipping)\n"
        f"  {len(english_pending)} English rows — will copy transcript through\n"
        f"  {len(needs_translation)} non-English rows — will translate on GPU\n"
        f"  {skipped_empty} empty transcripts (skipped)\n"
        f"  {skipped_no_lang} rows with no language code (run detect_language.py first)",
        flush=True,
    )
 
    # Copy English transcripts through immediately — no model needed.
    if english_pending:
        for idx, text in english_pending:
            df.at[idx, TRANS_COL] = text
        df.to_excel(EXCEL_FILE, index=False)
        print(f"  → English passthrough done, saved {len(english_pending)} rows.\n", flush=True)
 
    if not needs_translation:
        print("Nothing left to translate. Done.")
        return
 
    # ── Load NLLB only if there's actual work to do ──────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {NLLB_MODEL} on {device} …", flush=True)
    if device.type == "cuda":
        print(
            f"  CUDA device: {torch.cuda.get_device_name(0)} "
            f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)",
            flush=True,
        )
 
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
    nllb = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",   # faster attention on Ampere+ (RTX 3070)
    ).to(device)
    nllb.eval()
    print("Model ready.\n", flush=True)
 
    # ── Translate ────────────────────────────────────────────────────────
    total = len(needs_translation)
    processed = 0
    t_start = time.time()
 
    for n, (idx, lang_code, text) in enumerate(needs_translation, start=1):
        if _stop:
            break
 
        df.at[idx, TRANS_COL] = translate(tokenizer, nllb, device, text, lang_code)
        processed += 1
 
        elapsed = time.time() - t_start
        rate    = processed / elapsed if elapsed > 0 else 1
        eta_min = (total - processed) / rate / 60
 
        preview = str(df.at[idx, TRANS_COL])[:70].replace("\n", " ")
        print(
            f"[{n:5d}/{total}] row {idx+1} [{lang_code}] | {preview!r}"
            f"  (~{eta_min:.0f} min left)",
            flush=True,
        )
 
        if processed % SAVE_EVERY == 0:
            df.to_excel(EXCEL_FILE, index=False)
            print(f"  → checkpoint saved.", flush=True)
 
    df.to_excel(EXCEL_FILE, index=False)
    elapsed = time.time() - t_start
    print(
        f"\nDone. Translated {processed} rows in {elapsed/60:.1f} min. "
        f"File: {EXCEL_FILE}"
    )
 
 
if __name__ == "__main__":
    main()