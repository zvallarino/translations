#!/usr/bin/env python3
"""
Translate transcripts in urls.csv.
  - Language detection: FastText lid.176.bin (176 langs, incl. Hausa/Yoruba/Igbo)
  - Translation:        Meta NLLB-200 (1.3B or 3.3B) on GPU → English
 
Long transcripts are split into sentence-sized chunks before translation, since
NLLB has a hard 1024-token ceiling per call. Output chunks are joined back into
a single string per row.
 
Saves every SAVE_EVERY rows. Ctrl+C exits safely without losing progress.
Re-running skips rows already completed (both Language and Translation filled).
"""
 
import os, re, signal, time, urllib.request, warnings
warnings.filterwarnings("ignore")
 
import pandas as pd
import torch
import fasttext
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
 
# ── Config ────────────────────────────────────────────────────────────────
CSV_FILE      = "urls.csv"
FT_MODEL_PATH = "lid.176.bin"
FT_MODEL_URL  = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
 
# Pick ONE model. 1.3B is the recommended overnight default on an 8 GB 3070.
# 3.3B is the strongest NLLB; it needs ~6.5 GB VRAM at float16 and is ~2.5x
# slower per sentence. Comment/uncomment to switch.
NLLB_MODEL    = "facebook/nllb-200-1.3B"
# NLLB_MODEL  = "facebook/nllb-200-3.3B"
 
SAVE_EVERY    = 10
MAX_INPUT_TOKENS  = 400   # per chunk going into the model (NLLB hard limit is 1024)
MAX_OUTPUT_TOKENS = 512   # per chunk coming out
NUM_BEAMS         = 4     # beam search; quality bump on low-resource languages
 
# ── Graceful Ctrl+C ───────────────────────────────────────────────────────
_stop = False
def _on_sigint(sig, frame):
    global _stop
    print("\n[!] Ctrl+C — saving after this row then exiting.", flush=True)
    _stop = True
signal.signal(signal.SIGINT, _on_sigint)
 
# ── FastText code → human-readable language name ──────────────────────────
FT_CODE_TO_NAME = {
    "af": "Afrikaans",      "ak": "Akan",            "am": "Amharic",
    "ar": "Arabic",         "az": "Azerbaijani",      "be": "Belarusian",
    "bg": "Bulgarian",      "bm": "Bambara",          "bn": "Bengali",
    "bs": "Bosnian",        "ca": "Catalan",          "cs": "Czech",
    "cy": "Welsh",          "da": "Danish",           "de": "German",
    "el": "Greek",          "en": "English",          "eo": "Esperanto",
    "es": "Spanish",        "et": "Estonian",         "eu": "Basque",
    "fa": "Persian",        "ff": "Fula",             "fi": "Finnish",
    "fr": "French",         "fy": "Frisian",          "ga": "Irish",
    "gd": "Scottish Gaelic","gl": "Galician",         "gu": "Gujarati",
    "ha": "Hausa",          "he": "Hebrew",           "hi": "Hindi",
    "hr": "Croatian",       "ht": "Haitian Creole",   "hu": "Hungarian",
    "hy": "Armenian",       "id": "Indonesian",       "ig": "Igbo",
    "is": "Icelandic",      "it": "Italian",          "ja": "Japanese",
    "jv": "Javanese",       "ka": "Georgian",         "ki": "Kikuyu",
    "kk": "Kazakh",         "km": "Khmer",            "kn": "Kannada",
    "ko": "Korean",         "ku": "Kurdish",          "ky": "Kyrgyz",
    "lg": "Luganda",        "ln": "Lingala",          "lo": "Lao",
    "lt": "Lithuanian",     "lv": "Latvian",          "mg": "Malagasy",
    "mi": "Maori",          "mk": "Macedonian",       "ml": "Malayalam",
    "mn": "Mongolian",      "mr": "Marathi",          "ms": "Malay",
    "mt": "Maltese",        "my": "Burmese",          "nb": "Norwegian",
    "ne": "Nepali",         "nl": "Dutch",            "nn": "Norwegian Nynorsk",
    "om": "Oromo",          "or": "Odia",             "pa": "Punjabi",
    "pl": "Polish",         "ps": "Pashto",           "pt": "Portuguese",
    "qu": "Quechua",        "ro": "Romanian",         "ru": "Russian",
    "rw": "Kinyarwanda",    "sd": "Sindhi",           "si": "Sinhala",
    "sk": "Slovak",         "sl": "Slovenian",        "sn": "Shona",
    "so": "Somali",         "sq": "Albanian",         "sr": "Serbian",
    "ss": "Swati",          "st": "Southern Sotho",   "su": "Sundanese",
    "sv": "Swedish",        "sw": "Swahili",          "ta": "Tamil",
    "te": "Telugu",         "tg": "Tajik",            "th": "Thai",
    "ti": "Tigrinya",       "tl": "Filipino",         "tn": "Tswana",
    "tr": "Turkish",        "ts": "Tsonga",           "tt": "Tatar",
    "tw": "Twi",            "ug": "Uyghur",           "uk": "Ukrainian",
    "ur": "Urdu",           "uz": "Uzbek",            "ve": "Venda",
    "vi": "Vietnamese",     "wo": "Wolof",            "xh": "Xhosa",
    "yi": "Yiddish",        "yo": "Yoruba",           "zh": "Chinese",
    "zu": "Zulu",
}
 
# FastText 2-letter code → NLLB language+script code
FT_TO_NLLB = {
    "af": "afr_Latn",  "ak": "aka_Latn",  "am": "amh_Ethi",  "ar": "arb_Arab",
    "az": "azj_Latn",  "be": "bel_Cyrl",  "bg": "bul_Cyrl",  "bm": "bam_Latn",
    "bn": "ben_Beng",  "bs": "bos_Latn",  "ca": "cat_Latn",  "cs": "ces_Latn",
    "cy": "cym_Latn",  "da": "dan_Latn",  "de": "deu_Latn",  "el": "ell_Grek",
    "en": "eng_Latn",  "eo": "epo_Latn",  "es": "spa_Latn",  "et": "est_Latn",
    "eu": "eus_Latn",  "fa": "pes_Arab",  "ff": "fuv_Latn",  "fi": "fin_Latn",
    "fr": "fra_Latn",  "ga": "gle_Latn",  "gl": "glg_Latn",  "gu": "guj_Gujr",
    "ha": "hau_Latn",  "he": "heb_Hebr",  "hi": "hin_Deva",  "hr": "hrv_Latn",
    "ht": "hat_Latn",  "hu": "hun_Latn",  "hy": "hye_Armn",  "id": "ind_Latn",
    "ig": "ibo_Latn",  "is": "isl_Latn",  "it": "ita_Latn",  "ja": "jpn_Jpan",
    "jv": "jav_Latn",  "ka": "kat_Geor",  "ki": "kik_Latn",  "kk": "kaz_Cyrl",
    "km": "khm_Khmr",  "kn": "kan_Knda",  "ko": "kor_Hang",  "ku": "kmr_Latn",
    "ky": "kir_Cyrl",  "lg": "lug_Latn",  "ln": "lin_Latn",  "lo": "lao_Laoo",
    "lt": "lit_Latn",  "lv": "lvs_Latn",  "mg": "plt_Latn",  "mi": "mri_Latn",
    "mk": "mkd_Cyrl",  "ml": "mal_Mlym",  "mn": "khk_Cyrl",  "mr": "mar_Deva",
    "ms": "zsm_Latn",  "mt": "mlt_Latn",  "my": "mya_Mymr",  "nb": "nob_Latn",
    "ne": "npi_Deva",  "nl": "nld_Latn",  "om": "gaz_Latn",  "or": "ory_Orya",
    "pa": "pan_Guru",  "pl": "pol_Latn",  "ps": "pbt_Arab",  "pt": "por_Latn",
    "ro": "ron_Latn",  "ru": "rus_Cyrl",  "rw": "kin_Latn",  "sd": "snd_Arab",
    "si": "sin_Sinh",  "sk": "slk_Latn",  "sl": "slv_Latn",  "sn": "sna_Latn",
    "so": "som_Latn",  "sq": "als_Latn",  "sr": "srp_Cyrl",  "ss": "ssw_Latn",
    "st": "sot_Latn",  "su": "sun_Latn",  "sv": "swe_Latn",  "sw": "swh_Latn",
    "ta": "tam_Taml",  "te": "tel_Telu",  "tg": "tgk_Cyrl",  "th": "tha_Thai",
    "ti": "tir_Ethi",  "tl": "tgl_Latn",  "tn": "tsn_Latn",  "tr": "tur_Latn",
    "ts": "tso_Latn",  "tt": "tat_Cyrl",  "ug": "uig_Arab",  "uk": "ukr_Cyrl",
    "ur": "urd_Arab",  "uz": "uzn_Latn",  "ve": "ven_Latn",  "vi": "vie_Latn",
    "wo": "wol_Latn",  "xh": "xho_Latn",  "yi": "ydd_Hebr",  "yo": "yor_Latn",
    "zh": "zho_Hans",  "zu": "zul_Latn",
}
 
 
def download_with_progress(url: str, dest: str) -> None:
    def _hook(count, block_size, total_size):
        pct = min(100, count * block_size * 100 // total_size)
        print(f"\r  {pct}% ({count * block_size // 1_000_000} MB / {total_size // 1_000_000} MB)",
              end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()
 
 
def detect_language(ft_model, text: str):
    """Return (iso2_code, full_name)."""
    clean = text.replace("\n", " ")[:500]
    labels, probs = ft_model.predict(clean, k=1)
    code = labels[0].replace("__label__", "")
    return code, FT_CODE_TO_NAME.get(code, f"Unknown ({code})")
 
 
# Sentence-ish splitter: ends on . ! ? or newline; also handles … and 。
# Simple regex is good enough for transcripts (no need for a full sentence
# tokenizer here — transcripts are messy and over-engineering this hurts).
_SENT_SPLIT = re.compile(r"(?<=[\.!\?…。])\s+|\n+")
 
 
def split_into_chunks(text: str, tokenizer, max_tokens: int) -> list[str]:
    """
    Split `text` into chunks where each chunk tokenizes to <= max_tokens.
    First splits on sentence boundaries, then greedily packs sentences
    into chunks. Sentences longer than max_tokens themselves are further
    split on whitespace as a fallback.
    """
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        return []
 
    def tok_len(s: str) -> int:
        # add_special_tokens=False so we don't double-count BOS/EOS per piece
        return len(tokenizer(s, add_special_tokens=False).input_ids)
 
    chunks = []
    current = []
    current_len = 0
 
    for sent in sentences:
        sent_len = tok_len(sent)
 
        # A single sentence that's too long → fall back to word-splitting it.
        if sent_len > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            words = sent.split()
            buf, buf_len = [], 0
            for w in words:
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
 
        outputs = []
        tgt_id = tokenizer.convert_tokens_to_ids("eng_Latn")
        for chunk in chunks:
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_TOKENS + 32,  # small safety margin
            ).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    forced_bos_token_id=tgt_id,
                    max_length=MAX_OUTPUT_TOKENS,
                    num_beams=NUM_BEAMS,
                    no_repeat_ngram_size=3,
                )
            outputs.append(tokenizer.decode(out_ids[0], skip_special_tokens=True))
        return " ".join(outputs).strip()
    except torch.cuda.OutOfMemoryError as exc:
        # If a chunk OOMs (rare with the chunker), free the cache and report.
        torch.cuda.empty_cache()
        return f"[cuda-oom: {exc}]"
    except Exception as exc:
        return f"[error: {exc}]"
 
 
def main():
    # ── Download FastText model if missing ───────────────────────────────
    if not os.path.exists(FT_MODEL_PATH):
        print(f"Downloading FastText language-ID model (~126 MB) …")
        download_with_progress(FT_MODEL_URL, FT_MODEL_PATH)
 
    print("Loading FastText model …", flush=True)
    ft = fasttext.load_model(FT_MODEL_PATH)
 
    # ── Load NLLB ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {NLLB_MODEL} on {device} …", flush=True)
    if device.type == "cuda":
        print(f"  CUDA device: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)",
              flush=True)
    tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
    nllb = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    nllb.eval()
    print("Models ready.\n", flush=True)
 
    # ── Load spreadsheet ─────────────────────────────────────────────────
    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.str.strip()
 
    TRANSCRIPT_COL = "Transcript"
    LANG_COL       = "Language_NLLB"
    TRANS_COL      = "Translation_NLLB"
 
    # Add new columns if they don't exist yet
    if LANG_COL not in df.columns:
        df[LANG_COL] = pd.NA
    if TRANS_COL not in df.columns:
        df[TRANS_COL] = pd.NA
 
    # Cast to object so strings can be assigned cleanly
    df[LANG_COL]  = df[LANG_COL].astype(object)
    df[TRANS_COL] = df[TRANS_COL].astype(object)
 
    total      = len(df)
    processed  = 0
    already_ok = 0
    t_start    = time.time()
 
    for idx in range(total):
        if _stop:
            break
 
        transcript = df.at[idx, TRANSCRIPT_COL]
        if pd.isna(transcript) or str(transcript).strip() == "":
            continue
 
        # Skip rows already completed
        if pd.notna(df.at[idx, LANG_COL]) and pd.notna(df.at[idx, TRANS_COL]):
            already_ok += 1
            continue
 
        text = str(transcript).strip()
 
        # Language detection
        try:
            lang_code, lang_name = detect_language(ft, text)
        except Exception as exc:
            lang_code, lang_name = "unk", f"Unknown (error: {exc})"
 
        df.at[idx, LANG_COL] = lang_name
 
        # Translation (copy as-is if English, translate otherwise)
        if lang_code == "en":
            df.at[idx, TRANS_COL] = text
        else:
            df.at[idx, TRANS_COL] = translate(tokenizer, nllb, device, text, lang_code)
 
        processed += 1
        elapsed  = time.time() - t_start
        rate     = processed / elapsed if elapsed > 0 else 1
        todo     = total - already_ok - processed
        eta_min  = todo / rate / 60
 
        preview = str(df.at[idx, TRANS_COL])[:70].replace("\n", " ")
        print(
            f"[{idx+1:4d}/{total}] {lang_name:20s} | {preview!r}"
            f"  (~{eta_min:.0f} min left)",
            flush=True,
        )
 
        if processed % SAVE_EVERY == 0:
            df.to_csv(CSV_FILE, index=False)
            print(f"  → checkpoint saved at row {idx+1}", flush=True)
 
    # Final save
    df.to_csv(CSV_FILE, index=False)
    elapsed = time.time() - t_start
    print(
        f"\nDone. {processed} rows translated, {already_ok} already complete."
        f"  Total: {elapsed/60:.1f} min.  File: {CSV_FILE}"
    )
 
 
if __name__ == "__main__":
    main()