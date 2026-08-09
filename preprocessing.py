"""
Modul preprocessing teks untuk Klasifikasi Sentimen Ulasan Pawon Mbah Gito.

Direplikasi persis dari alur di notebook (Copy_of_implementasiii.ipynb):
1. Case Folding
2. Cleansing (hapus URL, angka, tanda baca, kata 1 huruf)
3. Tokenizing
4. Normalisasi Slang (butuh file slang.csv, kolom: slang, formal)
5. Stopword Removal (stopword bahasa Indonesia NLTK, dikurangi kata-kata penting
   yang membawa makna sentimen seperti "tidak", "enak", "mahal", dll.)
6. Stemming (Sastrawi)
7. Join token -> string siap untuk TF-IDF
"""

import os
import re

import nltk
nltk.download('stopwords') # Jika pakai NLTK
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =========================================================
# SETUP (dijalankan sekali saat modul di-import)
# =========================================================

try:
    _all_stopwords = set(stopwords.words("indonesian"))
except LookupError:
    nltk.download("stopwords")
    _all_stopwords = set(stopwords.words("indonesian"))

# Kata-kata yang TIDAK dibuang meski ada di daftar stopword,
# karena membawa informasi sentimen penting.
_KATA_PENTING = {
    "tidak", "tak", "bukan", "belum",
    "jangan", "kurang", "ga", "gak",
    "enggak", "nggak", "ngga",
    "sangat", "sekali", "banget",
    "amat", "terlalu", "cukup",
    "agak", "lebih", "paling",
    "baik", "buruk", "bagus", "jelek",
    "enak", "lezat", "mantap", "nikmat",
    "ramah", "cepat", "lambat",
    "mahal", "murah", "bersih", "kotor",
    "puas", "kecewa",
}
ALL_STOPWORDS = _all_stopwords - _KATA_PENTING

_tokenizer = RegexpTokenizer(r"\w+")
_stemmer = StemmerFactory().create_stemmer()


def _load_slang_dict():
    """Muat kamus normalisasi slang dari slang.csv (kolom: slang, formal).

    Kalau file tidak ada, kembalikan dict kosong (normalisasi di-skip)
    supaya aplikasi tetap jalan tanpa file tersebut.
    """
    path = "slang.csv"
    if not os.path.exists(path):
        return {}
    try:
        import pandas as pd

        slang_df = pd.read_csv(path)
        return dict(zip(slang_df["slang"], slang_df["formal"]))
    except Exception:
        return {}


SLANG_DICT = _load_slang_dict()


# =========================================================
# TAHAPAN PREPROCESSING
# =========================================================

def clean_text(text):
    """Cleansing: hapus URL, karakter non-huruf, angka, spasi berlebih,
    dan kata dengan panjang 1 huruf."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(word for word in text.split() if len(word) > 1)
    return text


def tokenize(text):
    return _tokenizer.tokenize(text)


def normalisasi(tokens):
    return [SLANG_DICT[token] if token in SLANG_DICT else token for token in tokens]


def remove_stopwords(tokens):
    return [word for word in tokens if word not in ALL_STOPWORDS]


def stem_tokens(tokens):
    return [_stemmer.stem(token) for token in tokens]


def join_text(tokens):
    return " ".join(tokens)


# =========================================================
# FUNGSI UTAMA (dipakai oleh app.py)
# =========================================================

def preprocess_text_for_prediction(text):
    """Jalankan seluruh pipeline preprocessing pada satu teks ulasan
    dan kembalikan string bersih siap untuk divektorisasi TF-IDF."""

    # 1. Case Folding
    casefolded_text = str(text).lower()

    # 2. Cleansing
    cleaned_text = clean_text(casefolded_text)

    # 3. Tokenizing
    tokenized_text = tokenize(cleaned_text)

    # 4. Normalisasi Slang
    normalized_text = normalisasi(tokenized_text)

    # 5. Stopword Removal
    no_stopword_text = remove_stopwords(normalized_text)

    # 6. Stemming
    stemmed_text = stem_tokens(no_stopword_text)

    # 7. Gabungkan token untuk TF-IDF
    text_result = join_text(stemmed_text)

    return text_result
