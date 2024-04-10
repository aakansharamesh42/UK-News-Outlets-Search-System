import os
from enum import Enum

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CHILD_INDEX_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "index", "child")
)

GLOBAL_INDEX_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "index", "global")
)
MONOGRAM_PKL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "spell_checking_and_autocomplete_files",
        "symspell_dictionary.pkl",
    )
)

MONOGRAM_AND_BIGRAM_DICTIONARY_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "spell_checking_and_autocomplete_files",
        "monogram_and_bigram_dictionary.data",
    )
)

FULL_TXT_CORPUS_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "spell_checking_and_autocomplete_files",
        "corpus.txt",
    )
)

STOP_WORDS_FILE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),  # The directory of the constants.py file
        "ttds_2023_english_stop_words.txt",  # Assuming it's directly under utils
    )
)

QUERY_EXPANSION_MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "word2vec_files",
    )
)


class Source(Enum):
    BBC = "bbc"
    GBN = "gbn"
    IND = "ind"
    TELE = "tele"
