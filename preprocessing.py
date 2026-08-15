"""
Modul preprocessing teks untuk Klasifikasi Sentimen Ulasan Pawon Mbah Gito.

Tahapan preprocessing:
1. Case Folding
2. Cleansing
3. Tokenizing
4. Normalisasi Slang
5. Negation Handling
6. Stopword Removal
7. Stemming
8. Join token -> string siap untuk TF-IDF
"""

import os
import re

import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# =========================================================
# SETUP
# =========================================================

try:
    _all_stopwords = set(stopwords.words("indonesian"))
except LookupError:
    nltk.download("stopwords")
    _all_stopwords = set(stopwords.words("indonesian"))


# Kata-kata yang tidak dibuang karena memiliki makna sentimen
_KATA_PENTING = {
    # Negasi
    "tidak", "tak", "bukan", "belum",
    "jangan", "kurang", "ga", "gak",
    "enggak", "nggak", "ngga",

    # Intensitas
    "sangat", "sekali", "banget",
    "amat", "terlalu", "cukup",
    "agak", "lebih", "paling",

    # Penilaian umum
    "baik", "buruk", "bagus", "jelek",
    "enak", "lezat", "mantap", "nikmat",
    "puas", "kecewa", "suka", "menyesal",

    # Rasa makanan
    "gurih", "asin", "manis", "pedas",
    "pahit", "asam", "hambar", "tawar",
    "sedap", "segar", "nikmat",

    # Kualitas makanan
    "matang", "mentah", "hangat", "panas",
    "dingin", "fresh", "berkualitas",
    "porsi", "banyak", "sedikit",

    # Pelayanan
    "ramah", "cepat", "lambat",
    "pelayanan", "pelayan", "waiter",
    "kasir", "sigap", "sopan",
    "responsif", "menunggu", "tunggu",
    "lama", "antri", "antre",

    # Harga
    "mahal", "murah", "harga",
    "terjangkau", "worth", "worthit",
    "murahan",

    # Kebersihan
    "bersih", "kotor", "higienis",
    "bau", "wangi",

    # Suasana/tempat
    "nyaman", "tidaknyaman",
    "luas", "sempit", "ramai",
    "sepi", "berisik", "tenang",
    "indah", "bagus", "jelek",

    # Fasilitas
    "parkir", "toilet", "musala",
    "wifi", "fasilitas", "tempat",

    # Pengalaman pelanggan
    "recommended", "rekomendasi",
    "rekomen", "favorit", "cocok",
    "puas", "kecewa", "mengecewakan",
    "memuaskan", "menyenangkan",
    "menarik", "nyaman"
}

ALL_STOPWORDS = _all_stopwords - _KATA_PENTING

_tokenizer = RegexpTokenizer(r"\w+")
_stemmer = StemmerFactory().create_stemmer()


# =========================================================
# NEGATION HANDLING
# =========================================================

_NEGATION_WORDS = {
    "tidak",
    "tak",
    "bukan",
    "belum",
    "jangan",
    "ga",
    "gak",
    "enggak",
    "nggak",
    "ngga"
}


def handle_negation(tokens):
    """
    Menangani kata negasi dengan menggabungkan kata negasi
    dengan satu kata setelahnya.

    Contoh:
        tidak enak    -> tidak_enak
        tidak bagus   -> tidak_bagus
        nggak nyaman  -> nggak_nyaman
        bukan murah   -> bukan_murah
    """

    result = []
    i = 0

    while i < len(tokens):
        word = tokens[i]

        if word in _NEGATION_WORDS and i + 1 < len(tokens):
            next_word = tokens[i + 1]

            # Gabungkan negasi dengan kata setelahnya
            result.append(f"{word}_{next_word}")
            i += 2
        else:
            result.append(word)
            i += 1

    return result


# =========================================================
# SLANG DICTIONARY
# =========================================================

def _load_slang_dict():
    """Muat kamus normalisasi slang dari slang.csv."""

    path = "slang.csv"

    if not os.path.exists(path):
        return {}

    try:
        import pandas as pd

        slang_df = pd.read_csv(path)

        return dict(
            zip(
                slang_df["slang"],
                slang_df["formal"]
            )
        )

    except Exception:
        return {}


SLANG_DICT = _load_slang_dict()


# =========================================================
# TAHAPAN PREPROCESSING
# =========================================================

def clean_text(text):
    """Cleansing: hapus URL, karakter non-huruf, angka,
    spasi berlebih, dan kata dengan panjang 1 huruf."""

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\d+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    text = " ".join(
        word
        for word in text.split()
        if len(word) > 1
    )

    return text


def tokenize(text):
    return _tokenizer.tokenize(text)


def normalisasi(tokens):
    return [
        SLANG_DICT[token]
        if token in SLANG_DICT
        else token
        for token in tokens
    ]


def remove_stopwords(tokens):
    return [
        word
        for word in tokens
        if word not in ALL_STOPWORDS
    ]


def stem_tokens(tokens):
    return [
        _stemmer.stem(token)
        for token in tokens
    ]


def join_text(tokens):
    return " ".join(tokens)


# =========================================================
# FUNGSI UTAMA
# =========================================================

def preprocess_text_for_prediction(text):
    """
    Menjalankan seluruh pipeline preprocessing pada satu
    teks ulasan dan mengembalikan string siap TF-IDF.
    """

    # 1. Case Folding
    casefolded_text = str(text).lower()

    # 2. Cleansing
    cleaned_text = clean_text(casefolded_text)

    # 3. Tokenizing
    tokenized_text = tokenize(cleaned_text)

    # 4. Normalisasi Slang
    normalized_text = normalisasi(tokenized_text)

    # 5. Negation Handling
    #    Dilakukan sebelum stopword removal agar kata yang
    #    dilekatkan pada negasi (mis. "tidak" + kata berikutnya)
    #    tidak ikut terbuang sebagai stopword.
    negation_text = handle_negation(normalized_text)

    # 6. Stopword Removal
    no_stopword_text = remove_stopwords(negation_text)

    # 7. Stemming
    stemmed_text = stem_tokens(no_stopword_text)

    # 8. Gabungkan token untuk TF-IDF
    text_result = join_text(stemmed_text)

    return text_result
