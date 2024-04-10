import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from typing import List, Tuple
from gensim.models import Word2Vec
import pickle
from common import get_preprocessed_words
import os, sys
from constant import QUERY_EXPANSION_MODEL_PATH

class QueryExpander:
    """
    Class to expand a query using a word2vec model which can be trained on the
    given corpus format.
    """

    def __init__(
        self, model_path: str = "word2vec_200_10.model"
    ) -> None:
        self.model = Word2Vec.load(os.path.join(QUERY_EXPANSION_MODEL_PATH, model_path))
        self.words = self.model.wv.index_to_key
        self.vectors = self.model.wv.vectors

    def create_all_document_csv(
        self, data_path: str, outlet_folders: List[str], output_path: str
    ) -> None:
        """
        Create a single CSV file containing all the articles from the given outlet folders.
        """
        articles = []
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
                    for index, row in df.iterrows():
                        articles.append(
                            {"doc_id": row["doc_id"], "content": row["content"]}
                        )

        output_df = pd.DataFrame(articles)

        # save_output_csv
        output_df.to_csv(
            output_path,
            index=False,
        )

    def create_unstemmed_pickle_list(
        self, data_csv_path: str, output_path: str
    ) -> None:
        """
        Create a pickle file containing a list of preprocessed documents without stemming.
        """
        df = pd.read_csv(data_csv_path)

        processed_documents_unstemmed = []

        for index in tqdm(range(0, len(df["doc_id"]))):
            try:
                processed_documents_unstemmed.append(
                    get_preprocessed_words(df.iloc[index]["content"], stemming=False)
                )
            except:
                pass

        with open(output_path, "wb") as f:
            pickle.dump(processed_documents_unstemmed, f)

    def create_word2vec_model_from_pickle_list(
        self,
        data_pickle_path: str,
        output_folder_not_full_path: str,
        vector_size: int = 200,
        window_size: int = 10,
    ) -> None:
        """
        Create a word2vec model from a pickle file containing a list of preprocessed documents.

        Note - the output_folder_not_full_path should be the path to the folder
        where the model will be saved.
        """

        with open(data_pickle_path, "rb") as f:
            processed_documents_unstemmed = pickle.load(f)

        model = Word2Vec(
            processed_documents_unstemmed,
            vector_size=vector_size,
            window=window_size,
            min_count=1,
            workers=min(1, os.cpu_count() - 2),
        )

        model.save(
            output_folder_not_full_path
            + "/word2vec_"
            + str(vector_size)
            + "_"
            + str(window_size)
            + ".model"
        )

    def load_model(self, model_path: str) -> None:
        """
        Load a word2vec model from the given path.
        """
        self.model = Word2Vec.load(model_path)
        self.words = self.model.wv.index_to_key
        self.vectors = self.model.wv.vectors

    def expand_query(self, query: str, top_n: int = 3) -> Tuple[str, List[str]]:
        query_terms = get_preprocessed_words(query, stopping=True, stemming=False)
        expanded_query_terms = []
        preprocessed_terms_set = set()  # To store preprocessed versions for comparison

        for term in query_terms:
            preprocessed_term = get_preprocessed_words(
                term, stopping=True, stemming=True
            )
            if not preprocessed_terms_set.intersection(set(preprocessed_term)):
                expanded_query_terms.append(term)
                preprocessed_terms_set.update(preprocessed_term)

            try:
                similar_words = self.model.wv.most_similar(term, topn=top_n)
                for word, similarity in similar_words:
                    preprocessed_word = get_preprocessed_words(
                        word, stopping=True, stemming=True
                    )
                    if not preprocessed_terms_set.intersection(set(preprocessed_word)):
                        expanded_query_terms.append(word)
                        preprocessed_terms_set.update(preprocessed_word)
            except KeyError:
                # Skip if the word is not in the vocabulary
                pass

        added_terms = [term for term in expanded_query_terms if term not in query_terms]

        return " ".join(expanded_query_terms), added_terms
