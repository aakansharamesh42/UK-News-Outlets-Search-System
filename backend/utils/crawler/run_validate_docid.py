import pandas as pd
import os, sys
import asyncio
import orjson

from tqdm import tqdm

from datetime import datetime

FILENAME = os.path.basename(__file__)
BASEPATH = os.path.dirname(__file__)
UTILPATH = os.path.dirname(BASEPATH)

sys.path.append(BASEPATH)
sys.path.append(UTILPATH)

from redis_utils import get_doc_size
from common import Logger

def set_index(df_, start_index=0, col_target='doc_id'):
    cols = df_.columns
    df = df_.reset_index()
    df['index'] += start_index
    df.drop(col_target, axis=1, inplace=True)
    df.rename(columns={'index': col_target}, inplace=True)
    return df[cols]

def counter(full, batch, func, col, doc_ids):
    batch[col] = func(doc_ids)
    full[col] = func(batch[col], full[col])
    return full, batch

if __name__ == "__main__":
    logpath = os.path.join(UTILPATH, 'logger.log')
    logger = Logger(logpath)

    logger.log_event('info', f'{FILENAME} - Start script')


    is_check_index = True
    is_check_sentiment = False
    is_check_summary = False

    # today = datetime.now()
    today = datetime(2024, 3, 9)
    today_str = today.strftime("%Y%m%d")
    today_str_dash = today.strftime("%Y-%m-%d")

    start_index = asyncio.run(get_doc_size())
    logger.log_event('info', f'{FILENAME} - Start Index: {start_index}')

    folder_path = os.path.join(UTILPATH, "data", today_str)
    files = os.listdir(folder_path)
    files = [i for i in files if f'old_dt_{today_str}' in i]
    min_index = {
        'data': 1_000_000_000,
        'index': 1_000_000_000,
        'sentiment': 1_000_000_000,
        'summary': 1_000_000_000
    }
    max_index = {
        'data': 0,
        'index': 0,
        'sentiment': 0,
        'summary': 0
    }

    batch_min_index = {}
    batch_max_index = {}

    for idx, f in enumerate(files):
        datafile = os.path.join(folder_path, f)
        inputfile = f"data/{today_str}"
        datapath = os.path.join(folder_path, datafile)
        df = pd.read_csv(datapath, usecols=['doc_id'])
        
        max_index, batch_max_index = counter(max_index, batch_max_index, max, 'data', df['doc_id'])
        min_index, batch_min_index = counter(min_index, batch_min_index, min, 'data', df['doc_id'])
        
        if is_check_index:
            # Check Index
            indexname = f.replace('old_dt_', 'index_').replace('.csv', '.json')
            indexpath = os.path.join(folder_path, indexname)

            with open(indexpath, 'rb') as f_:
                index = orjson.loads(f_.read())
            doc_ids_index = index['meta']['doc_ids_list']

            max_index, batch_max_index = counter(max_index, batch_max_index, max, 'index', doc_ids_index)
            min_index, batch_min_index = counter(min_index, batch_min_index, min, 'index', doc_ids_index)
            is_valid_index = min_index['data'] == min_index['index']
            assert is_valid_index, f"Error: is_valid_index"

        if is_check_sentiment:
            # Check Sentiment
            sentimentfile = f.replace('old_dt_', 'sentiment_').replace('.csv', '.json')
            sentimentpath = os.path.join(folder_path, sentimentfile)
            with open(sentimentpath, 'rb') as f_:
                sentiment_json = orjson.loads(f_.read())
            doc_ids_sentiment = [int(i) for i in sentiment_json.keys()]
        
            max_index, batch_max_index = counter(max_index, batch_max_index, max, 'sentiment', doc_ids_sentiment)
            min_index, batch_min_index = counter(min_index, batch_min_index, min, 'sentiment', doc_ids_sentiment)

            is_valid_sentiment = min_index['data'] == min_index['sentiment']
            assert is_valid_sentiment, f"Error: is_valid_sentiment"
            
        if is_check_summary:
            # Check Summary
            summaryfile = f.replace('old_dt_', 'summary_').replace('.csv', '.json')
            summarypath = os.path.join(folder_path, summaryfile)
            with open(summarypath, 'rb') as f_:
                summary_json = orjson.loads(f_.read())
            doc_ids_summary = [int(i) for i in summary_json.keys()]

            max_index, batch_max_index = counter(max_index, batch_max_index, max, 'summary', doc_ids_summary)
            min_index, batch_min_index = counter(min_index, batch_min_index, min, 'summary', doc_ids_summary)
        
            is_valid_summary = min_index['data'] == min_index['summary']
            assert is_valid_summary, f"Error: is_valid_summary"

    is_continuous = min_index['index'] == start_index
    print(is_continuous, start_index, min_index['index'])
    print(f"is_check_index: {is_check_index}")
    print(f"is_check_sentiment: {is_check_sentiment}")
    print(f"is_check_summary: {is_check_summary}")