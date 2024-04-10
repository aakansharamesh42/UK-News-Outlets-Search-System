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

today = datetime.now()
today_str = today.strftime("%Y%m%d")
today_str_dash = today.strftime("%Y-%m-%d")
folder_path = os.path.join(UTILPATH, "data", today_str)

inputfile = os.path.join(folder_path, f"data_{today_str}.csv")
outputfile = os.path.join(folder_path, f"sentiment_dictionary.json")



data = {}
cats = []
data_2 = {}
for idx, row in df.iterrows():
    url = row['url']
    doc_id = row['doc_id']

    topic = "-".join(url.split("/")[3:-1])

    data[doc_id] = topic
    data_2[topic] = doc_id
    cats.append(topic)