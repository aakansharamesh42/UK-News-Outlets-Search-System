import orjson
import os
import sys
import asyncio

# from tqdm import tqdm

BASEPATH = os.path.dirname(__file__)
sys.path.append(BASEPATH)

# from redis_utils import get_redis_config, update_doc_size, batch_push
from common import read_binary_file
from basetype import InvertedIndex
from redis_utils import initialize_async_redis, update_index, get_redis_config, update_tfidf_index
from constant import CHILD_INDEX_PATH
from build_index import merge_inverted_indices
from typing import Tuple, Dict

def load_index(path_index="result/inverted_index.json"):
    with open(path_index, "r+") as f:
        data_json = orjson.loads(f.read())
    return data_json


def process_dict_in_batches(input_dict, batch_size, prefix="w:"):
    """Deprecated soon"""
    keys = list(input_dict.keys())
    num_keys = len(keys)
    batches = []
    for i in range(0, num_keys, batch_size):
        batch_keys = keys[i : i + batch_size]
        batch = {prefix + key: str(input_dict[key]) for key in batch_keys}
        batches.append(batch)
    return batches

def get_tf(inverted_index: InvertedIndex) -> Dict[str, Dict[str, int]]:
    tf_index = {}
    for term, doc_ids in inverted_index.index.items():
        for doc_id, positions in doc_ids.items():
            if term not in tf_index:
                tf_index[term] = {}
            tf_index[term][doc_id] = len(positions)
    return tf_index

async def push_inverted_indices_to_redis(batch_size=10):
    files = os.listdir(CHILD_INDEX_PATH)
    file_batches = [files[i:i+batch_size] for i in range(0, len(files), batch_size)]
    for idx, batch in enumerate(file_batches):
        # first file
        parent_inverted_index = InvertedIndex.model_validate_json(read_binary_file(os.path.join(CHILD_INDEX_PATH, batch[0])))
        for f in batch[1:]:
            child_inverted_index = InvertedIndex.model_validate_json(read_binary_file(os.path.join(CHILD_INDEX_PATH, f)))
            merge_inverted_indices(parent_inverted_index.index, child_inverted_index.index)
            parent_inverted_index.meta.document_size += child_inverted_index.meta.document_size
            parent_inverted_index.meta.doc_ids_list.extend(child_inverted_index.meta.doc_ids_list)
        
        await update_index(parent_inverted_index)
        print(f"\r{' '*100}\r IDX: {idx+1}/{len(file_batches)} for positional inverted index", end="")
        await update_tfidf_index(parent_inverted_index)
        print(f"\r{' '*100}\r IDX: {idx+1}/{len(file_batches)} for tfidf index", end="")
        
        # free memory
        del parent_inverted_index
            

if __name__ == "__main__":
    # files = os.listdir(CHILD_INDEX_PATH)
    
    # for idx, f in enumerate(files):
    #     filepath = os.path.join(CHILD_INDEX_PATH, f)
    #     inverted_index_str = read_binary_file(filepath)
    #     inverted_index = InvertedIndex.model_validate_json(inverted_index_str)
    #     asyncio.run(update_index(inverted_index))
    #     print(f"\r{' '*100}\r IDX: {idx}", end="")
    asyncio.run(push_inverted_indices_to_redis(10))