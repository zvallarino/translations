#!/usr/bin/env python3
"""
Step 1 of 2 — Language detection only.
 
Reads `urls.xlsx`, looks at the `Transcript` column, runs FastText lid.176 on
each non-empty row, and writes two new columns:
 
    Language_NLLB       — human-readable name ("English", "Urdu", …)
    Language_NLLB_Code  — 2-letter code ("en", "ur", …) for downstream filtering
 
This script is CPU-only and very fast — FastText handles thousands of rows
per minute. Run this first, then use translate_to_english.py for step 2.
 
Re-running is safe: rows that already have BOTH language columns filled in
are skipped. Ctrl-C is safe (saves on exit).
"""
 
import os
import signal
import sys
import time
import urllib.request
import warnings
 
warnings.filterwarnings("ignore")
 
import pandas as pd
import fasttext
 
from lang_codes import detect_language
 
# ── Config ────────────────────────────────────────────────────────────────
EXCEL_FILE    = "urls.xlsx"
FT_MODEL_PATH = "lid.176.bin"
FT_MODEL_URL  = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
 
TRANSCRIPT_COL = "Transcript"
LANG_COL       = "Language_NLLB"        # human-readable name
LANG_CODE_COL  = "Language_NLLB_Code"   # 2-letter ISO code
 
SAVE_EVERY = 500   # detection is fast — checkpoint every 500 rows
 
 
# ── Graceful Ctrl-C ───────────────────────────────────────────────────────
_stop = False
def _on_sigint(sig, frame):
    global _stop
    print("\n[!] Ctrl-C — saving after this row then exiting.", flush=True)
    _stop = True
signal.signal(signal.SIGINT, _on_sigint)
 
 
def download_with_progress(url: str, dest: str) -> None:
    def _hook(count, block_size, total_size):
        pct = min(100, count * block_size * 100 // total_size)
        print(
            f"\r  {pct}% ({count * block_size // 1_000_000} MB / "
            f"{total_size // 1_000_000} MB)",
            end="", flush=True,
        )
    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()
 
 
def main():
    if not os.path.exists(EXCEL_FILE):
        sys.exit(f"[error] {EXCEL_FILE} not found in current directory.")
 
    if not os.path.exists(FT_MODEL_PATH):
        print(f"Downloading FastText language-ID model (~126 MB) …")
        download_with_progress(FT_MODEL_URL, FT_MODEL_PATH)
 
    print("Loading FastText model …", flush=True)
    ft = fasttext.load_model(FT_MODEL_PATH)
 
    print(f"Reading {EXCEL_FILE} …", flush=True)
    df = pd.read_excel(EXCEL_FILE)
 
    df.columns = df.columns.str.strip()
 
    if TRANSCRIPT_COL not in df.columns:
        sys.exit(
            f"[error] No '{TRANSCRIPT_COL}' column in {EXCEL_FILE}. "
            f"Columns found: {list(df.columns)}"
        )
 
    # Add columns if missing; ensure dtype is object so we can store strings
    # in rows that previously held NaN.
    for col in (LANG_COL, LANG_CODE_COL):
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].astype(object)
 
    total      = len(df)
    processed  = 0
    skipped    = 0
    already_ok = 0
    t_start    = time.time()
 
    print(f"Processing {total} rows.\n", flush=True)
 
    for idx in range(total):
        if _stop:
            break
 
        transcript = df.at[idx, TRANSCRIPT_COL]
        if pd.isna(transcript) or str(transcript).strip() == "":
            skipped += 1
            continue
 
        # Skip rows already labeled
        if pd.notna(df.at[idx, LANG_COL]) and pd.notna(df.at[idx, LANG_CODE_COL]):
            already_ok += 1
            continue
 
        text = str(transcript).strip()
        try:
            lang_code, lang_name = detect_language(ft, text)
        except Exception as exc:
            lang_code, lang_name = "unk", f"Unknown (error: {exc})"
 
        df.at[idx, LANG_COL]      = lang_name
        df.at[idx, LANG_CODE_COL] = lang_code
        processed += 1
 
        # Light progress every 100 rows so we can see it's alive
        if processed % 100 == 0:
            elapsed = time.time() - t_start
            rate    = processed / elapsed if elapsed > 0 else 1
            print(
                f"[{idx+1:5d}/{total}] {processed} done @ {rate:.0f} rows/sec "
                f"(latest: {lang_name})",
                flush=True,
            )
 
        if processed > 0 and processed % SAVE_EVERY == 0:
            df.to_excel(EXCEL_FILE, index=False)
            print(f"  → checkpoint saved at row {idx+1}", flush=True)
 
    # Final save
    df.to_excel(EXCEL_FILE, index=False)
 
    elapsed = time.time() - t_start
    print(
        f"\nDone. Detected language on {processed} rows, "
        f"{already_ok} already had language, {skipped} skipped (empty transcript). "
        f"Took {elapsed:.1f}s. File: {EXCEL_FILE}"
    )
 
    # Summary distribution so you can decide what to translate
    if processed > 0 or already_ok > 0:
        labeled = df[df[LANG_COL].notna()]
        if len(labeled) > 0:
            print("\nLanguage distribution:")
            counts = labeled[LANG_COL].value_counts()
            for name, count in counts.items():
                print(f"  {name:30s} {count:6d}")
 
 
if __name__ == "__main__":
    main()