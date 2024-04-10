import json, re, os, sys
import asyncio

import multiprocessing

n_cpu = multiprocessing.cpu_count()


BASEPATH = os.path.dirname(__file__)
sys.path.append(BASEPATH)


from module_past_data import scrape_old, scrape_old_async

url_target = "https://www.independent.co.uk/sitemap.xml"
n_pages = 0

if __name__ == "__main__":
    asyncio.run(
        scrape_old_async(
            url_target=url_target,
            n_workers=6,
            max_days=180,
            batch_size=300,
            output_path="data/ind/ind_data_{}_{}.csv",
            n_pages=n_pages,
            debug=False,
        )
    )
