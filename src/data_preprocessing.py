import re
import string
import unicodedata
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

pd.options.mode.chained_assignment = None

seed = 0
np.random.seed(seed)

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

current_dir = Path(__file__).parent
project_root = current_dir.parent

DICTIONARIES_PATH = project_root / "data" / "dictionaries"


def cleaningText(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # hapus link
    text = re.sub(r"\d+", "", text)  # hapus angka
    text = text.replace("-", " ")  # hapus tanda hubung
    text = re.sub(r"[^\w\s]", " ", text)  # hapus tanda baca
    text = text.replace("²", "")  # hanya hapus karakter ²
    text = text.strip()  # hapus spasi di awal dan akhir
    text = text.translate(str.maketrans("", "", string.punctuation))  # hapus tanda baca
    text = re.sub(
        r"\s+", " ", text
    )  # hapus spasi lebih dari satu termasuk newline, tab
    return text


def tokenizingText(text):  # Memecah atau membagi string, teks menjadi daftar token
    text = text.split()
    return text


slang_words = pd.read_csv(DICTIONARIES_PATH / "colloquial-indonesian-lexicon.csv")
slang_words_dict = (
    slang_words[["slang", "formal"]].set_index("slang")["formal"].to_dict()
)


def replace_slang_word(tokens):
    return [slang_words_dict.get(token, token) for token in tokens]


def filteringText(text):  # Menghapus stopwords dalam teks
    list_stopwords = set(stopwords.words("indonesian"))
    list_stopwords1 = set(stopwords.words("english"))
    list_stopwords.update(list_stopwords1)
    list_stopwords.update(
        [
            "iya",
            "yaa",
            "gak",
            "nya",
            "na",
            "sih",
            "ku",
            "di",
            "ga",
            "ya",
            "gaa",
            "loh",
            "kah",
            "woi",
            "woii",
            "woy",
            "yg",
            "yang",
        ]
    )

    kata_penting = [
        # sentimen umum
        "enak",
        "enggak",
        "ga",
        "tidak",
        "buruk",
        "baik",
        "bagus",
        "murah",
        "mahal",
        "cepat",
        "lambat",
        "nyaman",
        "kotor",
        "bersih",
        "ramah",
        "dingin",
        "panas",
        "sejuk",
        "gelap",
        "terang",
        "sangat",
        "kurang",
        "banget",
        "sekali",
        "parah",
        "biasa",
        "lumayan",
        "jelek",
        "sempurna",
        # aspek makanan
        "makanan",
        "minuman",
        "rasa",
        "porsi",
        "menu",
        "varian",
        "pedas",
        "manis",
        "asin",
        "gurih",
        "kopi",
        "teh",
        "jus",
        "roti",
        "nasi",
        "mie",
        "dessert",
        "sambal",
        # aspek harga
        "harga",
        "terjangkau",
        "diskon",
        "promo",
        "mahal",
        "murah",
        "worth",
        "sepadan",
        # aspek pelayanan
        "pelayanan",
        "staff",
        "waiter",
        "kasir",
        "ramah",
        "sopan",
        "cepat",
        "lama",
        "menunggu",
        "responsif",
        # aspek tempat / suasana
        "tempat",
        "lokasi",
        "parkir",
        "wifi",
        "fasilitas",
        "kursi",
        "meja",
        "toilet",
        "akses",
        "suasana",
        "tenang",
        "rame",
        "berisik",
        "hujan",
        "atap",
        "indoor",
        "outdoor",
        "nongkrong",
        # ekspresi pengguna
        "rekomendasi",
        "kecewa",
        "senang",
        "suka",
        "kurang",
        "cocok",
        "tidak",
        "kapok",
        "ulang",
        "lagi",
        "sering",
    ]

    stopword_ekspresi = [
        "wkwkwkwkwkwkwk",
        "wkwkwk",
        "wkwwk",
        "wkwk",
        "wk",
        "hehe",
        "he",
        "hihi",
        "hoho",
        "haha",
        "hadeh",
        "huhuhuu",
        "huhuhu",
        "huh",
        "hmm",
        "hmmm",
        "hmmmm",
        "eh",
        "yaelah",
        "yah",
        "ya",
        "yaa",
        "yg",
        "yang",
        "loh",
        "lah",
        "deh",
        "dong",
        "nih",
        "cie",
        "ciee",
        "cieee",
        "wle",
        "weleh",
        "anjay",
        "anjir",
        "anjrit",
        "eaa",
        "ea",
        "loh",
        "lucu",
        "lol",
        "ngakak",
        "wew",
        "ih",
        "issh",
        "ish",
        "cieeee",
        "aaaaaaaaaaa",
        "aaaaaaa",
        "aaa",
        "aa",
        "a",
    ]

    list_stopwords = list(list_stopwords - set(kata_penting))
    list_stopwords = set(list_stopwords + stopword_ekspresi)

    filtered = []
    for txt in text:
        if txt not in list_stopwords:
            filtered.append(txt)
    text = filtered
    return text


def stemmingText(
    text,
):  # Mengurangi kata ke bentuk dasarnya yang menghilangkan imbuhan awalan dan akhiran atau ke akar kata
    # Membuat objek stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # Memecah teks menjadi daftar kata
    words = text.split()

    # Menerapkan stemming pada setiap kata dalam daftar
    stemmed_words = [stemmer.stem(word) for word in words]

    # Menggabungkan kata-kata yang telah distem
    stemmed_text = " ".join(stemmed_words)

    return stemmed_text


def toSentence(list_words):  # Mengubah daftar kata menjadi kalimat
    sentence = " ".join(word for word in list_words)
    return sentence
