import re
import pandas as pd
import numpy as np
from collections import Counter
from math import log
import orjson
import warnings
import pandas as pd
from tqdm import tqdm
from typing import List
import os, sys

BASEPATH = os.path.dirname(__file__)
sys.path.append(BASEPATH)

from common import get_preprocessed_words

def compute_tf(text: str) -> dict:
    """Calculate term frequency for a given text."""
    tf_text = Counter(text)
    for i in tf_text:
        tf_text[i] = tf_text[i] / float(len(text))
    return tf_text


# Helper function to calculate inverse document frequency
def compute_idf(word: str, corpus: list) -> float:
    """Calculate inverse document frequency for a given word in a corpus."""
    return log(len(corpus) / sum([1.0 for i in corpus if word in i]))


# Calculate cosine similarity between title and each sentence
def cosine_similarity(vector1, vector2) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vector1, vector2)
    norm_a = np.linalg.norm(vector1)
    norm_b = np.linalg.norm(vector2)
    if norm_a == 0 or norm_b == 0:  # Avoid division by zero
        return 0.0
    return dot_product / (norm_a * norm_b)


def get_summary_sentence(
    title: str, content: str, number_of_initial_sentences_to_skip: int
) -> str:
    """Get the most relevant sentence from the article content setences based on the title."""
    split_regex = r"[.!?]"
    article_sentences = re.split(split_regex, content)
    article_sentences_lower = [x.lower() for x in article_sentences if x]

    # Preprocess title and sentences
    preprocessed_title = " ".join(get_preprocessed_words(title))
    preprocessed_sentences = [
        " ".join(get_preprocessed_words(sentence))
        for sentence in article_sentences_lower
    ]

    # Combine title and adjusted sentences for manual TF-IDF vectorization
    texts = [preprocessed_title] + preprocessed_sentences[
        number_of_initial_sentences_to_skip:
    ]

    # Create a set of all unique words
    vocabulary = set(word for text in texts for word in text.split())

    # Calculate TF for each document
    tfs = [compute_tf(text.split()) for text in texts]

    # Calculate IDF for each word in the vocabulary
    idfs = {word: compute_idf(word, texts) for word in vocabulary}

    # Calculate TF-IDF vectors
    tfidf_vectors = []
    for tf in tfs:
        tfidf_vectors.append(
            np.array([tf.get(word, 0) * idfs[word] for word in vocabulary])
        )

    # Compute similarities
    similarities = [
        cosine_similarity(tfidf_vectors[0], vec) for vec in tfidf_vectors[1:]
    ]

    # Find the most similar sentence index
    most_similar_sentence_index = (
        np.argmax(similarities) + number_of_initial_sentences_to_skip
    )
    return article_sentences[most_similar_sentence_index]


def get_summaries_of_csv_file(
    csv_file_path, summaries_dictionary=None, number_of_initial_sentences_to_skip=2
) -> dict:
    csv_dataframe = pd.read_csv(csv_file_path, usecols=['doc_id', 'title', 'content'])
    title_series = csv_dataframe["title"]
    content_series = csv_dataframe["content"]
    doc_id_series = csv_dataframe["doc_id"]

    if summaries_dictionary is None:
        summaries_dictionary = {}

    for i in range(len(title_series)):

        current_title = title_series[i]
        current_content = content_series[i]
        current_doc_id = doc_id_series[i]

        if str(current_doc_id) in summaries_dictionary.keys():
            warnings.warn(
                f"Duplicate doc_id found: {current_doc_id}. Overwriting the previous entry!"
            )

        try:
            current_summary = get_summary_sentence(
                current_title, current_content, number_of_initial_sentences_to_skip
            )
            summaries_dictionary[str(current_doc_id)] = current_summary.strip()

        except Exception as e:
            summaries_dictionary[str(current_doc_id)] = None
            continue

    return summaries_dictionary


def process_directories_and_write_summary_dictionary(
    data_path: str,
    outlet_folders: List[str],
    output_file_path: str,
    summaries_dictionary: dict = None,
    number_of_initial_sentences_to_skip: int = 2,
) -> dict:
    if summaries_dictionary is None:
        summaries_dictionary = {}

    for outlet_folder in outlet_folders:
        current_outlet_path = os.path.join(data_path, outlet_folder)
        current_csv_files = [
            file for file in os.listdir(current_outlet_path) if file.endswith(".csv")
        ]

        for csv_file in tqdm(
            current_csv_files, desc=f"Processing summaries for {outlet_folder}"
        ):
            current_csv_file_path = os.path.join(current_outlet_path, csv_file)
            summaries_dictionary = get_summaries_of_csv_file(
                current_csv_file_path,
                summaries_dictionary,
                number_of_initial_sentences_to_skip,
            )

    with open(output_file_path, "wb") as file:
        file.write(orjson.dumps(summaries_dictionary))

    print(f"Summaries written to {output_file_path}")

    return summaries_dictionary


if __name__ == "__main__":
    data_path = "C:/Users/Asus/Desktop/ttds-proj/backend/data"
    # outlet_folders = ["tele"]
    outlet_folders = ["bbc_new"]
    output_file_path = "summaries_files/summaries_2.json"

    process_directories_and_write_summary_dictionary(
        data_path,
        outlet_folders,
        output_file_path,
        summaries_dictionary=None,
        number_of_initial_sentences_to_skip=2,
    )