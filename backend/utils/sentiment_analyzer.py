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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


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


def get_sentiment_dictionary_from_csv_path(
    csv_path,
    device=DEVICE,
    model=MODEL,
    tokenizer=TOKENIZER,
    csv_sentiment_dictionary=None,
):
    """
    Returns {doc_id: [prob_negative, prob_neutral, prob_positive]}.

    If csv_sentiment_dictionary is None, a new dictionary will be created.
    """
    csv_dataframe = pd.read_csv(csv_path)
    content_series = csv_dataframe["content"]
    doc_id_series = csv_dataframe["doc_id"]

    model.to(device)

    if csv_sentiment_dictionary is None:
        csv_sentiment_dictionary = {}

    for index, text in enumerate(content_series):
        sentiment_list = analyze_sentiment(text, model, tokenizer, device)
        doc_id = doc_id_series[index]

        if str(doc_id) in csv_sentiment_dictionary.keys():
            warnings.warn(
                f"Duplicate doc_id found: {doc_id}. Overwriting the previous entry!"
            )

        csv_sentiment_dictionary[str(doc_id)] = sentiment_list

    return csv_sentiment_dictionary


if __name__ == "__main__":
    data_path = "C:/Users/Asus/Desktop/ttds-proj/backend/data/"
    outlet_folders = ["bbc", "gbn", "ind", "tele"]
    output_file_path = "sentiment_dictionary/sentiment_dictionary.json"
    sentiment_dictionary = {}

    # Iterate over each outlet folder
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
                try:
                    # Read the current CSV file into a pandas DataFrame
                    sentiment_dictionary = get_sentiment_dictionary_from_csv_path(
                        file_path, csv_sentiment_dictionary=sentiment_dictionary
                    )
                except Exception as e:
                    print(f"Error with {file_path}: {e}")

    with open(
        output_file_path, "wb"
    ) as file:  # 'wb' mode because orjson.dumps() returns bytes
        file.write(orjson.dumps(sentiment_dictionary))
