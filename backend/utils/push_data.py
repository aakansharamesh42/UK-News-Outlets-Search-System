import re
import os, sys
import asyncio

BASEPATH = os.path.dirname(__file__)
sys.path.append(BASEPATH)

from common import load_batch_from_news_source, get_indices_for_news_data
from redis_utils import batch_push_news_data
from constant import Source, DATA_PATH
from datetime import date
from typing import List, Tuple

async def do_push_data(
    # source: Source,
    # date: date,
    source_date_list: List[Tuple[Source, date]],
    interval=1,
):  
    for source, date in source_date_list:
        # file name format: {source_name}_{YYYY-MM-DD}_{start_number}_{end_number}.json
        time_str = date.strftime("%Y-%m-%d")
        pattern = re.compile(f"{source.value}_{time_str}_([0-9]+)_([0-9]+).csv")
        child_index_file_list = [
            file for file in os.listdir(DATA_PATH) if pattern.match(file)
        ]
        last_index = -1
        for file in child_index_file_list:
            # split by .csv
            file_name = file.split(".")[0]
            # split by _
            file_info = file_name.split("_")
            if int(file_info[-1]) > last_index:
                last_index = int(file_info[-1])

        indices = get_indices_for_news_data(source.value, date)

        # prune the indices
        indices = [index for index in indices if index > last_index]

        # divide the indices into intervals
        indices_batches = [
            indices[i : i + interval] for i in range(0, len(indices), interval)
        ]
        
        for idx, indices_batch in enumerate(indices_batches):
            news_batch = load_batch_from_news_source(
                source, date, indices_batch[0], indices_batch[-1]
            )
            await batch_push_news_data(news_batch)
            # print(f"\IDX: {idx}", end="", flush=True)


if __name__ == "__main__":
    asyncio.run(do_push_data([
        (Source.BBC, date(2024, 2, 17)),
        (Source.BBC, date(2024, 3, 4)),
        (Source.BBC, date(2024, 3, 9)),
        (Source.IND, date(2024, 2, 18)),
        (Source.GBN, date(2024, 2, 18)),
        (Source.TELE, date(2024, 2, 16))
    ]))
        
        
