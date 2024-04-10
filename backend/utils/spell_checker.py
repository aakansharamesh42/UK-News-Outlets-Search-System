from typing import List
from collections import Counter
import re
from symspellpy import SymSpell, Verbosity
import os
import pandas as pd
from tqdm import tqdm


class SpellChecker:
    def __init__(
        self, dictionary_path: str = "spell_checking_files/symspell_dictionary.pkl"
    ) -> None:

        self.sym_spell = SymSpell()
        if dictionary_path:
            self.sym_spell.load_pickle(dictionary_path)

    def create_spell_checking_txt(
        self,
        data_path: str,
        outlet_folders: List[str],
        output_corpus_path: str,
        corpus_from_scratch: bool = False,
    ) -> None:
        """
        From txts with "content" and "doc_id" columns, create a txt file with all the content from all articles.

        data_path: str
            The path to the directory containing the outlet folders.
        outlet_folders: list
            A list of the outlet folder names.
        corpus_from_scratch: bool
            If True, the output file will be overwritten if it exists; if False, content will be appended to the existing file.
        """
        print("Creating txt corpus of contents of all documents...")

        file_mode = "w" if corpus_from_scratch else "a"

        for outlet_folder in outlet_folders:
            # Construct the path to the current outlet folder
            folder_path = os.path.join(data_path, outlet_folder)
            # List all files in the current outlet folder
            all_file_paths = os.listdir(folder_path)

            # Iterate over each file in the current outlet folder
            for file_name in tqdm(all_file_paths, desc=outlet_folder):
                # Construct the full path to the current file
                file_path = os.path.join(folder_path, file_name)
                # Ensure the file is a CSV before attempting to read it
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path)
                    content_series = df["content"]
                    doc_id_series = df["doc_id"]
                    for index, article in enumerate(content_series):
                        try:
                            with open(
                                output_corpus_path, file_mode, encoding="utf-8"
                            ) as file:
                                file.write(article + "\n")
                            file_mode = "a"
                        except TypeError as te:
                            if (
                                "unsupported operand type(s) for +: 'float' and 'str'"
                                in str(te)
                            ):
                                # If the specific TypeError is caught, you might want to handle it differently
                                # or simply pass to ignore it.
                                pass
                            else:
                                # Handle other TypeErrors that are not about concatenating float with str
                                print(
                                    f"Skipping article {doc_id_series[index]} due to TypeError: {te}"
                                )
                        except Exception as e:
                            # This catches all other exceptions and prints the error message.
                            print(
                                f"Skipping article {doc_id_series[index]} due to error: {e}"
                            )

    def create_filtered_corpus(
        self, corpus_txt_path, filtered_corpus_path, min_frequency
    ):
        """Create a new corpus file with words having a minimum frequency."""
        print("Reading corpus file...")
        with open(corpus_txt_path, "r", encoding="utf-8") as file:
            corpus = file.read().lower()
        print("Tokenizing corpus...")
        words = re.findall(r"\w+", corpus)

        print("Counting word frequencies...")
        word_counts = Counter(words)

        # Initialize tqdm progress bar
        tqdm.pandas()

        # Filter words by frequency using tqdm for progress indication
        filtered_words = [
            word
            for word in tqdm(words, desc="Filtering words")
            if word_counts[word] > min_frequency
        ]

        # Save filtered corpus with progress indication
        with open(filtered_corpus_path, "w", encoding="utf-8") as file:
            for word in tqdm(filtered_words, desc="Writing filtered corpus"):
                file.write(f"{word} ")

    def create_and_save_spellcheck_dictionary(
        self,
        corpus_txt_path: str,
        output_dictionary_path: str,
        save_more_frequent_than: int = 1,
    ) -> None:
        """
        Create and save a SymSpell dictionary from a corpus.

        corpus_txt_path: str
            The path to the corpus txt file.
        output_dictionary_path: str
            Where to save the output dictionary.

        """
        print("Reading in corpus...")
        # with open(corpus_txt_path, "r", encoding="utf-8") as file:
        #     corpus = file.read()
        print("Creating dictionary...")
        # self.sym_spell.create_dictionary(corpus)
        self.sym_spell.create_dictionary(corpus_txt_path)
        self.sym_spell.save_pickle(output_dictionary_path)
        print("Dictionary created and saved in ", output_dictionary_path)

    def load_dictionary(self, dictionary_path: str) -> None:
        """Load the SymSpell dictionary from a pickle file."""
        self.sym_spell.load_pickle(dictionary_path)

    def correct_query(
        self, text: str, max_edit_distance: int = 2, ignore_non_words: bool = True
    ) -> str:
        """
        Correct a query using the loaded SymSpell dictionary.

        text: str
            The query to correct.
        max_edit_distance: int
            The maximum edit distance to look for corrections.
        ignore_non_words: bool
            Whether to ignore non-words when correcting the query.

        """
        suggestions = self.sym_spell.lookup_compound(
            text, max_edit_distance=max_edit_distance, ignore_non_words=ignore_non_words
        )
        return suggestions[0].term if suggestions else text


def norvig_correction(word):
    "SLOW! Most probable spelling correction for word."

    def words(text):
        return re.findall(r"\w+", text.lower())

    WORDS = Counter(words(open("spell_checking_files/very_big_string.txt").read()))

    def known(words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in WORDS)

    def P(word, N=sum(WORDS.values())):
        "Probability of `word`."
        return WORDS[word] / N

    def candidates(word):
        "Generate possible spelling corrections for word."

        def edits1(word):
            "All edits that are one edit away from `word`."
            letters = "abcdefghijklmnopqrstuvwxyz"
            splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
            deletes = [L + R[1:] for L, R in splits if R]
            transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
            replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
            inserts = [L + c + R for L, R in splits for c in letters]
            return set(deletes + transposes + replaces + inserts)

        def edits2(word):
            "All edits that are two edits away from `word`."
            return (e2 for e1 in edits1(word) for e2 in edits1(e1))

        return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]

    return max(candidates(word), key=P)


# if __name__ == "__main__":
#     data_path = "C:/Users/Asus/Desktop/ttds-proj/backend/data/"
#     # outlet_folders = ["bbc", "gbn", "ind", "tele"]
#     # corpus_path = "C:/Users/Asus/Desktop/ttds-proj/backend/utils/corpus.txt"

#     # create_spell_checking_txt(data_path, outlet_folders, output_corpus_path)

#    # filtered_corpus_path = "C:/Users/Asus/Desktop/ttds-proj/backend/utils/filtered_corpus.txt"

#     # create_filtered_corpus(corpus_path, filtered_corpus_path, 100)

#     # output_dictionary_path = "C:/Users/Asus/Desktop/ttds-proj/backend/utils/symspell_dictionary.pkl"

#     # create_spell_checker_dictionary(filtered_corpus_path, output_dictionary_path)

#     # pickle_file_path = "spell_checking_files/huge_symspell_dictionary.pkl"
#     # symspell_instance = load_sym_spell_instance("pickle_file_path")
#     # print(correct_query("helo", symspell_instance))
