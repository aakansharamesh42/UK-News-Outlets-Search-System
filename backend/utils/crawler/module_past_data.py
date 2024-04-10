import pandas as pd
import numpy as np
import requests
import xmltodict
import json, re, os, sys
import asyncio

import csv   

from bs4 import BeautifulSoup as soup
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

FILENAME = os.path.basename(__file__)
BASEPATH = os.path.dirname(__file__)
UTILPATH = os.path.dirname(BASEPATH)

from utils_crawler import req_get, get_link_and_date
from module_new_data import (
    run_scrape,
    get_new_data,
    filter_data_link,
    get_doc_size,
    Logger,
)


def get_list_of_pages(source="bbc", n_pages=1, outputdict='bbc_dict.csv'):
    list_pages = []
    if any([i in source for i in ["bbc", "telegraph", "gbnews"]]):
        indices = list(range(50, n_pages)) #1: 0-50, 2: 50-n_pages
        if "bbc" in source:
            indices.reverse()
            url_target = "https://www.bbc.com/sitemaps/https-sitemap-com-archive-{}.xml"
        elif "gbnews" in source:
            url_target = "https://www.gbnews.com/feeds/sitemaps/sitemap_{}.xml"
        else:
            url_target = "https://www.telegraph.co.uk/news/page-{}/"

        for page in indices:
            url_target_temp = url_target.format(page)
            list_pages.append(url_target_temp)

    elif "independent" in source:
        url_indep_a = "https://www.independent.co.uk/sitemap.xml"
        r = requests.get(url_indep_a)
        data = xmltodict.parse(r.content)

        data_link_raw = data["sitemapindex"]["sitemap"]
        list_pages = [
            i["loc"] for i in data_link_raw if "/sitemap-articles-" in i["loc"]
        ]
    else:
        list_pages = []

    data_links = []
    for page in tqdm(list_pages, desc="listing-pages"):
        if "telegraph" in source:
            r = req_get(page)
            data_url_target = soup(r.content)
            data_link = get_link_and_date(
                data_url_target, prefix="https://www.telegraph.co.uk"
            )
            data_links += data_link
        else:
            r = req_get(page)
            data = xmltodict.parse(r.content)
            data_link = data["urlset"]["url"]
            data_links += data_link

        df = pd.DataFrame(data_link)
        df.to_csv(
            outputdict,
            mode="a",
            header=not os.path.exists(outputdict),
            index=False,
        )
    return data_links


async def scrape_old_async(iter_size, batch_size, output_path, outputdict, start_doc_id, debug=False):
    logpath = os.path.join(UTILPATH, "logger.log")
    logger = Logger(logpath)

    all_data = []
    it = 0
    today = datetime.now()
    today_str = today.strftime("%Y%m%d")

    debug_max_doc = 10
    use_existing_dict = os.path.exists(outputdict)
    if not use_existing_dict:
        logger.log_event("info", f"{FILENAME} - Collecting pages")
        data_links = get_list_of_pages(source="bbc", n_pages=105, outputdict=outputdict)
    else:
        logger.log_event("info", f"{FILENAME} - Loading output dict")
        data_links = pd.read_csv(outputdict)

    logger.log_event("info", f"{FILENAME} - Filtering Link")
    data_link = filter_data_link(data_links, is_df=use_existing_dict)

    # Check links on redis
    logger.log_event("info", f"{FILENAME} - Filtering New Data Only")
    data_link_new = get_new_data(data_link, start_doc_id=start_doc_id)
    data_link_new = np.array(data_link_new)

    n_pages = len(data_link_new)
    logger.log_event("info", f"{FILENAME} - Total New Link: {n_pages}")

    all_data = []
    logger.log_event("info", f"{FILENAME} - Iterating")
    for i in tqdm(range(0, n_pages, iter_size), desc="Iterate:"):
        pkg = data_link_new[i : i + iter_size]

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

    logpath = os.path.join(UTILPATH, "logger.log")
    logger = Logger(logpath)

    logger.log_event("info", f"{FILENAME} - Start script")

    today = datetime.now()
    today_str = today.strftime("%Y%m%d")

    folder_path = os.path.join(UTILPATH, "data", today_str)

    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    outputfile = os.path.join(folder_path, "data_{}_{}.csv")
    outputdict = os.path.join(folder_path, f"dict_{today_str}.csv")

    start_doc_id = asyncio.run(get_doc_size())
    logger.log_event("info", f"{FILENAME} - Start Index: {start_doc_id}")

    # test_func(batch_size=5, output_path=outputfile, start_doc_id=start_doc_id, debug=False)
    asyncio.run(
        scrape_old_async(
            iter_size=10,
            batch_size=1000, output_path=outputfile, outputdict=outputdict, start_doc_id=start_doc_id, debug=False
        )
    )
    logger.log_event("info", f"{FILENAME} - Scrapping new data finished")
