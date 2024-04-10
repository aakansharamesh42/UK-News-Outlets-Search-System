import json, re, os, sys
import asyncio

import multiprocessing

n_cpu = multiprocessing.cpu_count()


BASEPATH = os.path.dirname(__file__)
sys.path.append(BASEPATH)


from module_past_data import scrape_old, scrape_old_async

url_bbc_a = "https://www.bbc.com/sitemaps/https-sitemap-com-archive-{}.xml"
n_pages = 105

if __name__ == "__main__":
    asyncio.run(
        scrape_old_async(
            url_target=url_bbc_a,
            n_pages=n_pages,
            n_workers=6,
            is_reverse=True,
            max_days=360,
            batch_size=500,
            output_path="data/bbc_data_{}_{}.csv",
            debug=False,
        )
    )
