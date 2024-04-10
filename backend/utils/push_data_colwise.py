import re
import os, sys
import asyncio
import orjson

BASEPATH = os.path.dirname(__file__)
sys.path.append(BASEPATH)

from redis_utils import set_news_data_col
from constant import Source, DATA_PATH
from datetime import date
from basetype import RedisKeys, RedisDocKeys
import numpy as np
import asyncio

from tqdm import tqdm

async def do_gather_task_push_value(json_data, key:RedisDocKeys, func):
    keys = list(json_data.keys())
    batch_size = 200
    for i in tqdm(range(0, len(keys), batch_size)):
        batch_keys = keys[i:i + batch_size]
        tasks = []
        for doc_id in batch_keys:
            doc_id_key = RedisKeys.document(doc_id)
            value = func(json_data[doc_id])
            tasks.append(set_news_data_col(doc_id_key, key, value))
        await asyncio.gather(*tasks)

async def do_push_value(path_to_json: str, key:RedisDocKeys, func):    
    with open(path_to_json, 'r+') as f:
        json_data = orjson.loads(f.read())

    await do_gather_task_push_value(json_data, key, func)

def func_summary(value):
    if value:
        return value.strip()
    else:
        return "Unable to get summary."

# Deprecated
# def func_sentiment(value):
#     labels = ['negative', 'neutral', 'positive']
#     return labels[np.argmax(value)]
    
def func_sentiment(value):
    if value is None:
        value = [0, 1, 0]
    res = [f"negative:{value[0]}", f"neutral:{value[1]}", f"positive:{value[2]}"]
    return orjson.dumps(res)

if __name__ == "__main__":
    path_to_sentiment = "data/sentiment_dictionary.json"
    # path_to_summary = "data/summaries.json"


    asyncio.run(do_push_value(path_to_sentiment, RedisDocKeys.sentiment, func_sentiment))
    # asyncio.run(do_push_value(path_to_summary, RedisDocKeys.summary, func_summary))
