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


BASEPATH = os.path.dirname(__file__)
sys.path.append(BASEPATH)

from utils_crawler import (
    get_content,
    get_hyper_text_str,
    get_figcaption_str,
    get_title,
    req_get,
    run_scrape,
)


def scrape_old(
    url_target, n_pages, is_reverse, max_days, batch_size, output_path, debug=False
):
    indices = list(range(1, n_pages))
    all_data = []
    it = 0
    today = datetime.now()
    today_str = today.strftime("%Y%m%d")

    debug_max_doc = 10

    if is_reverse:
        indices.reverse()

    for page in indices:
        url_target = url_target.format(page)
        r = req_get(url_target)
        data = xmltodict.parse(r.content)

        data_link = data["urlset"]["url"]
        source = url_target.split("/")[2]

        for target in tqdm(data_link, desc=f"page-{page}"):
            doc_url = target["loc"]
            doc_date = target["lastmod"]
            age = today - pd.to_datetime(doc_date[:10])
            age = age.days
            if age > max_days:
                continue

            if "bbc" in source:
                if not (
                    ("https://www.bbc.com/news/" in doc_url)
                    or ("https://www.bbc.com/newsround/" in doc_url)
                    or ("https://www.bbc.com/sport/" in doc_url)
                    or ("https://www.bbc.com/weather/" in doc_url)
                ):
                    continue

            r_doc = req_get(doc_url)
            soup_data = soup(r_doc.text, "html.parser")

            try:
                doc_title = get_title(soup_data)
            except:
                continue
            content = get_content(soup_data)
            hyper_text_str = get_hyper_text_str(soup_data)
            figcaption_text = get_figcaption_str(soup_data)

            if not doc_date:
                continue

            all_data.append(
                {
                    "source": source,
                    "title": doc_title,
                    "date": doc_date,
                    "content": content,
                    "hypertext": hyper_text_str,
                    "figcaption": figcaption_text,
                    "url": doc_url,
                }
            )

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

        if age > max_days:
            continue


def filter_data_link(data_links, source, today, max_days):
    if "bbc" in source:
        data_link = []
        for l in data_links:
            doc_url = l["loc"]
            if not (
                ("https://www.bbc.com/news/" in doc_url)
                or ("https://www.bbc.com/newsround/" in doc_url)
                or ("https://www.bbc.com/sport/" in doc_url)
                or ("https://www.bbc.com/weather/" in doc_url)
            ):
                continue
            else:
                data_link.append(l)
    else:
        data_link = data_links

    data_link_clean = []

    max_age = 0

    for l in data_link:
        date_publish_str = l.get("lastmod", "2000-01-01")[:10]
        date_publish = datetime.strptime(date_publish_str, "%Y-%m-%d")

        age = today - date_publish
        age = age.days
        max_age = max(max_age, age)
        if age > max_days:
            continue

        data_link_clean.append((l["loc"], l["lastmod"][:10], source))

    return data_link_clean, max_age


def get_list_of_pages(source="bbc", n_pages=1):
    list_pages = []
    if any([i in source for i in ["bbc", "telegraph", "gbnews"]]):
        indices = list(range(1, n_pages))
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

    return list_pages


def get_link_and_date(soup_data, prefix="https://www.telegraph.co.uk"):
    link_date = []
    article_data = soup_data.findAll("article")
    for article_ in article_data:
        if not article_.time:
            continue
        href_ = prefix + article_.h2.a["href"]
        date_ = article_.time["datetime"][:10]

        link_date.append({"loc": href_, "lastmod": date_})
    return link_date


async def scrape_old_async(
    url_target, n_workers, max_days, batch_size, output_path, n_pages=1, debug=False
):
    all_data = []
    it = 0
    today = datetime.now()
    today_str = today.strftime("%Y%m%d")
    source = url_target.split("/")[2]

    debug_max_doc = 10

    page_links = get_list_of_pages(source, n_pages)

    # Update N Pages info after filtering
    n_pages = len(page_links)

    for page_idx, page in enumerate(page_links):
        if "telegraph" in source:
            r = req_get(page)
            data_url_target = soup(r.content)
            data_links = get_link_and_date(
                data_url_target, prefix="https://www.telegraph.co.uk"
            )
        else:
            r = req_get(page)
            data = xmltodict.parse(r.content)
            data_links = data["urlset"]["url"]

        data_link, max_age = filter_data_link(data_links, source, today, max_days)

        data_link_chunks = np.array_split(data_link, len(data_link) // n_workers)

        for pkg in tqdm(data_link_chunks, desc=f"Page: {page_idx}/{n_pages}"):
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

        if max_age > max_days:
            continue
