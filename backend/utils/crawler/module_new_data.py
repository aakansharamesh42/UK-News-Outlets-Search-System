import pandas as pd
import numpy as np
import requests
import xmltodict
import json, re, os, sys
import asyncio

from bs4 import BeautifulSoup as soup
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

FILENAME = os.path.basename(__file__)
BASEPATH = os.path.dirname(__file__)
UTILPATH = os.path.dirname(BASEPATH)

sys.path.append(BASEPATH)
sys.path.append(UTILPATH)

from utils_crawler import run_scrape
from redis_utils import get_doc_size, check_batch_urls_exist
from common import Logger


def get_new_data(urls: dict, start_doc_id: int):
    results = check_batch_urls_exist(urls)
    new_links = []
    it = start_doc_id
    for value, result in zip(urls.keys(), results):
        if not bool(result):
            source = value.split("/")[2]
            new_links.append((value, urls[value], source, it))
            it += 1
    return new_links


def filter_data_link(data_links, is_df = False):
    data_link_clean = dict()

    max_age = 0

    if not is_df:
        for l in data_links:
            doc_url = l["loc"]
            if "www.bbc.com" in doc_url:
                if not (
                    ("https://www.bbc.com/news/" in doc_url)
                    or ("https://www.bbc.com/newsround/" in doc_url)
                    or ("https://www.bbc.com/sport/" in doc_url)
                    or ("https://www.bbc.com/weather/" in doc_url)
                ):
                    continue

            data_link_clean[l["loc"]] = l["lastmod"][:10]
    else:
        for idx,l in data_links.iterrows():
            doc_url = l["loc"]
            if "www.bbc.com" in doc_url:
                if not (
                    ("https://www.bbc.com/news/" in doc_url)
                    or ("https://www.bbc.com/newsround/" in doc_url)
                    or ("https://www.bbc.com/sport/" in doc_url)
                    or ("https://www.bbc.com/weather/" in doc_url)
                ):
                    continue

            data_link_clean[l["loc"]] = l["lastmod"][:10]

    return data_link_clean


def get_list_of_pages():
    url_targets = [
        "https://www.bbc.com/sitemaps/https-sitemap-com-news-1.xml",
        "https://www.bbc.com/sitemaps/https-sitemap-com-news-2.xml",
        "https://www.bbc.com/sitemaps/https-sitemap-com-news-3.xml",
        "https://www.independent.co.uk/sitemaps/googlenews",
        "https://www.telegraph.co.uk/custom/daily-news/sitemap.xml",
        "https://www.gbnews.com/feeds/sitemaps/news_1.xml",
    ]

    list_pages = []
    for url_target in url_targets:
        r = requests.get(url_target)
        data = xmltodict.parse(r.content)

        list_pages += data["urlset"]["url"]

    return list_pages


async def scrape_new_async(batch_size, output_path, start_doc_id, debug=False):
    all_data = []
    it = 0
    today = datetime.now()
    today_str = today.strftime("%Y%m%d")

    debug_max_doc = 10

    data_links = get_list_of_pages()
    data_link = filter_data_link(data_links)

    # Check links on redis
    data_link_new = get_new_data(data_link, start_doc_id=start_doc_id)
    data_link_new = np.array(data_link_new)

    n_pages = len(data_link_new)

    all_data = []
    for i in tqdm(range(0, n_pages, batch_size), desc="Iterate:"):
        pkg = data_link_new[i : i + batch_size]

        output_data = await run_scrape(pkg)
        all_data.extend(output_data)

        if debug:
            debug_max_doc -= 1
            batch_size = 5

            if debug_max_doc <= 0:
                debug_max_doc = 10
                break

        if len(all_data) >= batch_size:
            data_fin = pd.DataFrame(all_data)
            output_path_file = output_path.format(today_str, it)

            data_fin.to_csv(
                output_path_file,
                mode="a",
                header=not os.path.exists(output_path_file),
                index=False,
            )
            all_data = []
            it += 1
            if debug:
                exit()


if __name__ == "__main__":

    logpath = os.path.join(UTILPATH, 'logger.log')
    logger = Logger(logpath)

    logger.log_event('info', f'{FILENAME} - Start script')

    today = datetime.now()
    today_str = today.strftime("%Y%m%d")

    folder_path = os.path.join(UTILPATH, "data", today_str)

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    outputfile = os.path.join(folder_path, f"data_{today_str}.csv")

    start_doc_id = asyncio.run(get_doc_size())
    logger.log_event('info', f'{FILENAME} - Start Index: {start_doc_id}')

    asyncio.run(
        scrape_new_async(
            batch_size=5, output_path=outputfile, start_doc_id=start_doc_id, debug=False
        )
    )
    logger.log_event('info', f'{FILENAME} - Scrapping new data finished')
