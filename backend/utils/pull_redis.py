import os
import sys
import asyncio
from redis_utils import get_val, get_doc_size, get_doc_ids_list

from redis_utils import get_doc_data, get_doc_fields, get_docs_fields
from basetype import RedisDocKeys

if __name__ == "__main__":
    """To demonstrate collecting the data from redis"""
    # Get Index from DB 0
    word = "w:men"

    index_result = get_val(word)
    # document_size = get_doc_size()
    # doc_ids_list = get_doc_ids_list()
    # document_size = asyncio.run(get_doc_size())

    print(index_result)
    # print(len(doc_ids_list))
    # print(document_size)

    # Get data from DB 1
    doc_id = 123

    # Pull data from doc id
    # data_ = asyncio.run(get_doc_data(doc_id=doc_id))
    # print(data_)


    # Pull url from doc id
    # url_ = asyncio.run(get_docs_fields(doc_ids=[1,2], fields_list=[RedisDocKeys.url, RedisDocKeys.title]))
    # print(url_)

    # Pull title from doc id
    # title_ = asyncio.run(get_doc_key(doc_id=doc_id, key_list=[RedisDocKeys.title]))
    # print(title_)
