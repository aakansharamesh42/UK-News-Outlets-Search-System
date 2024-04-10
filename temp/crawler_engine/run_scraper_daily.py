import pandas as pd
import numpy as np
import requests
import xmltodict
import json, re, os

from bs4 import BeautifulSoup as soup
from tqdm import tqdm
from collections import defaultdict


BASEPATH = os.path.dirname(__file__)
sys.path.append(BASEPATH)

from utils import get_content, get_hyper_text_str, get_figcaption_str


debug = False
batch_size = 100
output_path = "data-2.csv"

all_data = []

url_targets = [
    "https://news.sky.com/sitemap/sitemap-news.xml",
    "https://www.bbc.com/sitemaps/https-sitemap-com-news-1.xml",
    "http://www.theguardian.com/sitemaps/news.xml",
    "https://www.independent.co.uk/sitemaps/googlenews",
    "https://www.telegraph.co.uk/custom/daily-news/sitemap.xml",
    "https://www.gbnews.com/feeds/sitemaps/news_1.xml",
]

for url_target in url_targets:
    r = requests.get(url_target)
    data = xmltodict.parse(r.content)
    raw_docs = data["urlset"]["url"]

    source = url_target.split("/")[2]
    n_doc = len(raw_docs)

    debug_max_doc = 10

    for idx_doc in tqdm(range(n_doc), desc=source):
        doc_url = data["urlset"]["url"][idx_doc]["loc"]
        doc_date = data["urlset"]["url"][idx_doc]["news:news"]["news:publication_date"][:10]
        doc_date = re.findall("(\d{4}-\d{2}-\d{2})", doc_date)[0]
        doc_title = data["urlset"]["url"][idx_doc]["news:news"]["news:title"]

        doc_title = re.sub("[(\s\s+)\n\t\"\']+", " ", doc_title).strip()

        if "bbc" in source:
            if not (
                ("https://www.bbc.com/news/" in doc_url)
                or ("https://www.bbc.com/newsround/" in doc_url)
                or ("https://www.bbc.com/sport/" in doc_url)
                or ("https://www.bbc.com/weather/" in doc_url)
            ):
                continue

        r_doc = requests.get(doc_url)

        soup_data = soup(r_doc.text, "html.parser")

        # Get Content
        content = get_content(soup_data)

        if not (content):
            continue

        # Get Hyptertext
        hyper_text_str = get_hyper_text_str(soup_data)

        # Get Image Caption
        figcaption_text = get_figcaption_str(soup_data)

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
            data_fin.to_csv(
                output_path,
                mode="a",
                header=not os.path.exists(output_path),
                index=False,
            )
            all_data = []
            if debug:
                exit()
