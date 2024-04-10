import pandas as pd
import os, sys
import asyncio

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

if __name__ == "__main__":
    logpath = os.path.join(UTILPATH, 'logger.log')
    logger = Logger(logpath)

    logger.log_event('info', f'{FILENAME} - Start script')

    today = datetime.now()
    today_str = today.strftime("%Y%m%d")
    today_str_dash = today.strftime("%Y-%m-%d")

    start_index = asyncio.run(get_doc_size())
    logger.log_event('info', f'{FILENAME} - Start Index: {start_index}')

    folder_path = os.path.join(UTILPATH, "data", today_str)
    files = os.listdir(folder_path)
    files = [i for i in files if f'data_{today_str}' in i]
    for idx, f in tqdm(enumerate(files)):
        new_file = os.path.join(folder_path, f)
        indexpath = f"data/{today_str}"
        old_name = f.replace('data_', 'old_dt_')

        old_path = os.path.join(folder_path, old_name)

        if not os.path.exists(old_path):
            os.rename(new_file, old_path)

        df = pd.read_csv(old_path)
        df = set_index(df, start_index=start_index)


        df.to_csv(new_file, index=False)
        start_index += len(df)
    
        logger.log_event('info', f'{FILENAME} - {idx} New Index: {start_index}')