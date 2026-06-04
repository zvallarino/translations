"""
Shared helpers for detect_language.py and translate_to_english.py.

Keeps the FastText code → human-readable name table and the FastText →
NLLB-200 code table in one place so the two scripts can't drift.
"""

# ── FastText 2-letter code → human-readable language name ────────────────
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

# ── FastText 2-letter code → NLLB language+script code ───────────────────
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


def detect_language(ft_model, text: str):
    """Return (iso2_code, full_name). Uses first 500 chars of cleaned text.

    500 chars is plenty for FastText to make a confident call — going longer
    just slows it down without improving accuracy on transcripts.
    """
    clean = text.replace("\n", " ")[:500]
    labels, probs = ft_model.predict(clean, k=1)
    code = labels[0].replace("__label__", "")
    return code, FT_CODE_TO_NAME.get(code, f"Unknown ({code})")
