### TO DO - MOVE THE MODEL, TOKENIZER, AND DEVICE DETERMINATION
# TO THE INITIALIZATION OF THE MODULE

import os
import warnings
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import orjson
import pandas as pd
import os, sys
import asyncio

from datetime import datetime

FILENAME = os.path.basename(__file__)
BASEPATH = os.path.dirname(__file__)
UTILPATH = os.path.dirname(BASEPATH)

sys.path.append(UTILPATH)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

from common import Logger
from basetype import RedisDocKeys
from push_data_colwise import do_gather_task_push_value, func_sentiment


def analyze_sentiment(text, model=MODEL, tokenizer=TOKENIZER, device=DEVICE):
    """
    Analyzes sentiment for a single piece of text and returns rounded sentiment probabilities.

    Parameters:
        text (str): The text to analyze.
        model: The pre-trained sentiment analysis model.
        tokenizer: The tokenizer for the model.
        device: The device to run the model on.

    Returns:
        list: A list containing the probabilities for [negative, neutral, positive] sentiments.
    """
    try:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        sentiments = (
            torch.nn.functional.softmax(outputs.logits, dim=-1)
            .to("cpu")[0]
            .float()
            .numpy()
        )
        # Round the first 2 elements and convert them to Python floats
        rounded_sentiments = [float(np.round(sentiment, 2)) for sentiment in sentiments]
        diff = 1.0 - sum(rounded_sentiments)
        rounded_sentiments[-1] = float(np.round(rounded_sentiments[-1] + diff, 2))
        return rounded_sentiments
    except Exception as e:
        return None


def get_sentiment_dictionary_from_df(
    df,
    device=DEVICE,
    model=MODEL,
    tokenizer=TOKENIZER,
    csv_sentiment_dictionary=None,
):
    """
    Returns {doc_id: [prob_negative, prob_neutral, prob_positive]}.

    If csv_sentiment_dictionary is None, a new dictionary will be created.
    """
    content_series = df["content"]
    doc_id_series = df["doc_id"]

    model.to(device)

    if csv_sentiment_dictionary is None:
        csv_sentiment_dictionary = {}

    for doc_id, text in zip(doc_id_series, content_series):
        sentiment_list = analyze_sentiment(text, model, tokenizer, device)

        if str(doc_id) in csv_sentiment_dictionary.keys():
            warnings.warn(
                f"Duplicate doc_id found: {doc_id}. Overwriting the previous entry!"
            )

        csv_sentiment_dictionary[str(doc_id)] = sentiment_list

    return csv_sentiment_dictionary


if __name__ == "__main__":
    logpath = os.path.join(UTILPATH, 'logger.log')
    logger = Logger(logpath)

    today = datetime.now()
    # today = datetime(2024, 3, 9)
    today_str = today.strftime("%Y%m%d")
    today_str_dash = today.strftime("%Y-%m-%d")
    folder_path = os.path.join(UTILPATH, "data", today_str)

    files = os.listdir(folder_path)
    files = [i for i in files if f'data_{today_str}' in i]

    # inputfile = os.path.join(folder_path, f"data_{today_str}.csv")

    for idx, f in tqdm(enumerate(files)):
        inputfile = os.path.join(folder_path, f)
        indexpath = f"data/{today_str}"
        indexname = f.replace('data_', 'sentiment_').replace('.csv', '.json')

        outputfile = os.path.join(folder_path, indexname)
        if os.path.exists(outputfile):
            # skip if the index does exist
            # with open(outputfile, "rb") as file:
            #     sentiment_dictionary = orjson.loads(file.read())
            #     # file.write(orjson.dumps(sentiment_dictionary))
            # logger.log_event('info', f'{FILENAME} - {idx} Updating the sentiment on Redis')
            # asyncio.run(do_gather_task_push_value(sentiment_dictionary, RedisDocKeys.sentiment, func_sentiment))
            continue
        print("No Thanks")

        logger.log_event('info', f'{FILENAME} - {idx} Start script')
        sentiment_dictionary = {}

        logger.log_event('info', f'{FILENAME} - {idx} Read Data in Chunk')
        df_all = pd.read_csv(inputfile, chunksize=100, usecols=["doc_id", "content"])

        # Iterate over each file in the current outlet folder
        logger.log_event('info', f'{FILENAME} - {idx} Iterating')
        for df in tqdm(df_all):
            # Read the current CSV file into a pandas DataFrame
            sentiment_dictionary = get_sentiment_dictionary_from_df(
                df, csv_sentiment_dictionary=sentiment_dictionary
            )
        logger.log_event('info', f'{FILENAME} - {idx} Dumping the data to json')

        with open(outputfile, "wb") as file:
            file.write(orjson.dumps(sentiment_dictionary))

        logger.log_event('info', f'{FILENAME} - {idx} Updating the sentiment on Redis')
        asyncio.run(do_gather_task_push_value(sentiment_dictionary, RedisDocKeys.sentiment, func_sentiment))

    logger.log_event('info', f'{FILENAME} - Done')