import json, re
import requests
import time

import asyncio
import aiohttp
import multiprocessing

from bs4 import BeautifulSoup as soup


def get_title(soup_data, source="bbc"):
    if source in ["bbc"]:
        title = soup_data.findAll("h1")
        if len(title) == 0:
            title = ""
        else:
            title = title[0].text
            title = re.sub("[(\s\s+)\t\"']+", " ", title)
            title = title.strip()
    else:
        title = ""

    return title


def get_content(soup_data):
    paragraph = [i.text for i in soup_data.findAll("p")]
    if len(paragraph) <= 3:
        return None
    content = " ".join(paragraph)
    content = re.sub(r"[(\s\s+)\t\"\“\”\’\'\‘]+", " ", content)
    return content


def get_hyper_text_str(soup_data):
    hyper_text_dict = {}
    for i in soup_data.findAll("p"):
        if i.a:
            key_ = re.sub("[(\s\s+)\n\t\"']+", " ", i.a.text.strip()).strip()
            val_ = i.a.get("href", "#NOLINK#")
            val_ = re.sub(r"[(\s\s+)\t\"\“\”\’\'\‘]+", " ", val_).strip()

            hyper_text_dict[key_] = val_
        else:
            pass
    hyper_text_str = json.dumps(hyper_text_dict)
    hyper_text_str = hyper_text_str.replace('"', "'")

    return hyper_text_str


def get_figcaption_str(soup_data):
    figcaption_dict = {}
    for idx, i in enumerate(soup_data.findAll("figcaption")):
        val_ = re.sub(r"[(\s\s+)\t\"\“\”\’\'\‘]+", " ", i.text).strip()
        figcaption_dict[idx] = val_
    figcaption_str = json.dumps(figcaption_dict)
    figcaption_str = figcaption_str.replace('"', "'")

    return figcaption_str


def get_link_and_date(soup_data, prefix="https://www.telegraph.co.uk"):
    link_date = []
    article_data = soup_data.findAll("article")
    for article_ in article_data:
        href_ = prefix + article_.h2.a["href"]
        date_ = article_.time["datetime"][:10]

        link_date.append({"loc": href_, "lastmod": date_})
    return link_date


def req_get(url, timeout=2, waittime=2):
    while True:
        try:
            r = requests.get(url, timeout=timeout)  # 10 seconds
            break
        except (requests.exceptions.Timeout, OSError):
            print("Timed out or OSError")
            time.sleep(waittime)
    return r


# Async IO
async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()


async def fetch_all_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)


def post_process(url, content_raw, date, source, doc_id):
    soup_data = soup(content_raw, features="lxml")

    doc_title = get_title(soup_data)
    content = get_content(soup_data)
    hyper_text_str = get_hyper_text_str(soup_data)
    figcaption_text = get_figcaption_str(soup_data)

    output = {
        "source": source,
        "title": doc_title,
        "date": date,
        "content": content,
        "hypertext": hyper_text_str,
        "figcaption": figcaption_text,
        "url": url,
        "doc_id":doc_id
    }
    return output


async def run_scrape(package):
    urls = package[:, 0]
    dates = package[:, 1]
    sources = package[:, 2]
    doc_id = package[:, 3]
    results = await fetch_all_urls(urls)

    # Post-processing in parallel using multiprocessing
    with multiprocessing.Pool() as pool:
        output = pool.starmap(post_process, zip(urls, results, dates, sources, doc_id))

    return output


# results = await fetch_all_urls(urls)
