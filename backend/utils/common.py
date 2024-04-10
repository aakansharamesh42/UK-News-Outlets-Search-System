import re
import orjson
import os
import pandas as pd
from nltk.stem import PorterStemmer
from xml.dom import minidom
from typing import List
from datetime import date
from constant import DATA_PATH, Source
from basetype import NewsArticlesFragment, NewsArticleData, NewsArticlesBatch
import numpy as np
import logging

# STOP_WORDS_FILE = "ttds_2023_english_stop_words.txt"
from constant import STOP_WORDS_FILE_PATH as STOP_WORDS_FILE

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

stemmer = PorterStemmer()
stop_words = None


def read_file(file_name: str, input_dir: str = "result") -> str:
    with open(os.path.join(CURRENT_DIR, input_dir, file_name), "r") as f:
        content = f.read()
    return content


def read_binary_file(file_name: str, input_dir: str = "result") -> bytes:
    with open(os.path.join(CURRENT_DIR, input_dir, file_name), "rb") as f:
        content = f.read()
    return content


def read_xml_file(file_name: str) -> minidom.Document:
    file = minidom.parse(file_name)
    return file


def get_stop_words(file_name: str = STOP_WORDS_FILE) -> list:
    assert os.path.exists(
        os.path.join(CURRENT_DIR, file_name)
    ), f"File {file_name} does not exist"
    with open(os.path.join(CURRENT_DIR, file_name), "r") as f:
        stop_words = f.read()
    return stop_words.split("\n")


def remove_stop_words(tokens: list) -> list:
    assert os.path.exists(
        os.path.join(CURRENT_DIR, STOP_WORDS_FILE)
    ), f"File {STOP_WORDS_FILE} does not exist"
    global stop_words
    if stop_words is None:
        stop_words = get_stop_words()
    return [token for token in tokens if token not in stop_words]


def tokenize(content: str) -> list:
    tokens = re.findall(r"\w+", content)
    # remove string that does not contain any english characters or digits
    tokens = [token for token in tokens if re.search(r"[a-zA-Z0-9]", token)]
    return tokens


def get_stemmed_words(tokens: list) -> list:
    words = [stemmer.stem(token) for token in tokens]
    return words


def replace_non_word_characters(content: str) -> str:
    # replace non word characters with space
    return re.sub(r"[^\w\s]", " ", content)


def get_preprocessed_words(
    content: str, stopping: bool = True, stemming: bool = True
) -> list:
    tokens = tokenize(content)
    # remove the phrase that does not contain any english characters or digits
    for token in tokens:
        if not re.search(r"[a-zA-Z0-9]", token):
            tokens.remove(token)
    tokens = [token.lower() for token in tokens]
    if stopping:
        tokens = remove_stop_words(tokens)
    if stemming:
        tokens = get_stemmed_words(tokens)
    return tokens


def save_json_file(file_name: str, data: dict, output_dir: str = "result"):
    if not os.path.exists(os.path.join(CURRENT_DIR, output_dir)):
        os.makedirs(os.path.join(CURRENT_DIR, output_dir))
    with open(os.path.join(CURRENT_DIR, output_dir, file_name), "wb") as f:
        f.write(orjson.dumps(data))


def load_json_file(file_name: str, input_dir: str = "result") -> dict:
    with open(os.path.join(CURRENT_DIR, input_dir, file_name), "rb") as f:
        data = f.read()
    return orjson.loads(data)


def get_indices_for_news_data(
    source_name: str,
    date: date,
) -> List[int]:
    # file name format: {source_name}_data_{date}_{number}.csv
    # date format: YYYYMMDD
    time_str = date.strftime("%Y%m%d")
    pattern = re.compile(f"{source_name}_data_{time_str}_([0-9]+).csv")

    data_path = os.path.join(DATA_PATH, source_name)
    assert os.path.exists(data_path), f"{data_path} does not exist"

    file_name_list = os.listdir(data_path)
    indices = []
    for file_name in file_name_list:
        match = pattern.match(file_name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def load_batch_from_news_source(
    source: Source,
    date: date,
    start_index: int = 0,
    end_index: int = -1,
) -> NewsArticlesBatch:

    indices = get_indices_for_news_data(source.value, date)
    assert start_index in indices, f"{start_index} is not in the indices list"

    end_index = len(indices) - 1 if end_index == -1 else end_index
    if end_index < start_index or end_index > len(indices):
        raise ValueError(f"Invalid end_index: {end_index}")

    indices = indices[indices.index(start_index) : indices.index(end_index) + 1]
    news_fragment_list = []
    doc_ids = []

    for index in indices:
        print(
            f"\r{' '*100}\rLoading {source.value} data {date.strftime('%Y%m%d')}_{index}.csv",
            end="",
        )
        filename = f"{source.value}_data_{date.strftime('%Y%m%d')}_{index}.csv"
        filepath = os.path.join(DATA_PATH, source.value, filename)
        df = pd.read_csv(filepath)
        df.fillna("", inplace=True)
        df["doc_id"] = df["doc_id"].astype(str)
        # convert the DataFrame to a list of dictionaries and change keys to lowercase
        news_article_list = []
        for row in df.itertuples(index=True):
            news_article = {k.lower(): v for k, v in row._asdict().items()}
            news_article_list.append(news_article)
            doc_ids.append(int(news_article["doc_id"]))
        # convert the list of dictionaries to NewsArticleData objects
        news_article_list = [
            NewsArticleData.model_validate(news_article)
            for news_article in news_article_list
        ]

        current_news_fragment = NewsArticlesFragment(
            source=source.value, date=date, index=index, articles=news_article_list
        )

        news_fragment_list.append(current_news_fragment)

    # add an new line
    print()

    return NewsArticlesBatch(
        doc_ids=sorted(doc_ids),
        indices={source.value: np.array(indices).astype(str).tolist()},
        fragments={source.value: news_fragment_list},
    )


def get_sources(datapath: str) -> List[str]:
    # check if the file exists
    if not os.path.exists(datapath):
        raise FileNotFoundError(f"{datapath} does not exist")

    # get the file list of current directory
    file_list = os.listdir(datapath)
    return file_list


def append_index_to_csv(source: Source, date: date, start_doc_id: int) -> int:
    # get the file list of current directory
    file_list = get_sources(os.path.join(DATA_PATH, source.value))
    indices = get_indices_for_news_data(source.value, date)
    current_doc_id = start_doc_id

    # read the csv file and append the index to the doc_id
    for index in indices:
        print(
            f"\r{' '*100}\rProcessing {source.value} data {date.strftime('%Y%m%d')}_{index}.csv",
            end="",
        )
        filename = f"{source.value}_data_{date.strftime('%Y%m%d')}_{index}.csv"
        filepath = os.path.join(DATA_PATH, source.value, filename)
        df = pd.read_csv(filepath)
        df.fillna("", inplace=True)
        # create column doc_id and append the current_doc_id to the doc_id
        df["doc_id"] = np.arange(current_doc_id, current_doc_id + len(df))
        df.to_csv(filepath, index=False)
        print(f"\r{' '*100}\r{filename} is processed", end="")

        current_doc_id += len(df)

    print()

    print("current_doc_id:", current_doc_id)
    # save the current_doc_id to a file
    with open("current_doc_id.txt", "w") as f:
        f.write(str(current_doc_id))

class Logger:
    def __init__(self, logfile):
        self.logfile = logfile
        self.logger = logging.getLogger('custom_logger')
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log_event(self, level, message):
        if level == 'debug':
            self.logger.debug(message)
        elif level == 'info':
            self.logger.info(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        elif level == 'critical':
            self.logger.critical(message)
        else:
            raise ValueError("Invalid log level")


if __name__ == "__main__":
    data = load_batch_from_news_source(Source.BBC, date(2024, 2, 17))
    # append_index_to_csv(Source.TELE, date= date(2024, 2, 16), start_doc_id=310427)
