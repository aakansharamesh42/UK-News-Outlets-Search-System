import json, re, os, sys
import asyncio

import multiprocessing

n_cpu = multiprocessing.cpu_count()


BASEPATH = os.path.dirname(__file__)
sys.path.append(BASEPATH)


from module_past_data import scrape_old, scrape_old_async

# url_target = "https://www.bbc.com/sitemaps/https-sitemap-com-archive-{}.xml"
# url_target = "https://www.independent.co.uk/sitemap.xml"
url_target = "https://www.telegraph.co.uk/news/page-{}/"
# url_target = "https://www.gbnews.com/feeds/sitemaps/sitemap_{}.xml"
# n_pages = 842
n_pages = 501  # 105

if __name__ == "__main__":
    asyncio.run(
        scrape_old_async(
            url_target=url_target,
            n_workers=2,
            max_days=3,
            batch_size=10,
            output_path="test/test_tele_data_{}_{}.csv",
            n_pages=n_pages,
            debug=True,
        )
    )
