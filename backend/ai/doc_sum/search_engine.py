import nltk
import pickle
from collections import Counter
from typing import List, NamedTuple, Dict
import math
import heapq
import os
import sys

index = {}
doc_lengths = {}
documents = {}

class Article(NamedTuple):
    title: str
    body: str

    def __repr__(self):
        if self.title != '' and self.body != '':
            return self.title + '\n' + self.body
        else:
            return self.title + self.body

def tokenize(text):
    return nltk.word_tokenize(text)


def preprocess(text):
    tokenized = tokenize(text.lower())
    return [w for w in tokenized if w.isalpha()]


# def build_index(docs: List[Article], paths: dict, dir: str):
def build_index(docs: Dict[int, Article], paths: dict, dir: str):
    global index, doc_lengths, documents
    index = {}
    doc_lengths = {}
    documents = {}

    print("Start building index...")
    processed = 0
    total = len(docs)
    tik = total / 100

    for doc_id, doc in docs.items():#enumerate(docs):
        if processed % tik == 0:
            perc = int(processed / total * 100)
            sys.stdout.write(f"\rDocuments processed - {perc}%")
            sys.stdout.flush()
        
        documents[doc_id] = doc

        doc_terms = preprocess(doc.title + doc.body)
        doc_lengths[doc_id] = len(doc_terms)

        tf = Counter()
        for term in doc_terms:
            tf[term] += 1
        
        for term in tf:
            if term in index:
                index[term][0] += 1
            else:
                index[term] = [1]
            index[term].append((doc_id, tf[term]))
        
        processed += 1

    if dir != '' and not os.path.exists(dir):
        os.makedirs(dir)
    
    with open(paths['index'], 'wb') as dump_file:
        pickle.dump(index, dump_file)
    
    with open(paths['lengths'], 'wb') as dump_file:
        pickle.dump(doc_lengths, dump_file)

    with open(paths['docs'], 'wb') as dump_file:
        pickle.dump(documents, dump_file)


def scores_val(query: Dict[str, int], doc_lengths: Dict[int, int], 
                  index, k1=1.2, b=0.75):
    
    scores = Counter()
    avgdl = sum(doc_lengths.values()) / len(doc_lengths)
    for term in query:
        if term in index and query[term] > 1e-15:
            idf = math.log10(len(doc_lengths) / (len(index[term]) - 1))
            for i in range(1, len(index[term])):
                doc_id, doc_freq = index[term][i]
                nominator = doc_freq * (k1 + 1)
                denominator = (doc_freq + k1 * (1 - b + b * doc_lengths[doc_id] / avgdl))
                scores[doc_id] += idf * nominator / denominator
    
    return dict(scores)


def answer_query(in_query, top_k, get_ids=False, is_raw=True, included_docs=None):
    if is_raw:
        query = preprocess(in_query)
        query = Counter(query)
    else:
        query = in_query
    
    scores = scores_val(query, doc_lengths, index)
    h = []
    _filter = included_docs is not None
    for doc_id in scores.keys():
        if _filter and (doc_id not in included_docs):
            continue
        neg_score = -scores[doc_id]
        heapq.heappush(h, (neg_score, doc_id))

    top_k_result = []
    top_k_ids = []
    top_k = min(top_k, len(h))
    for _ in range(top_k):
        best_so_far = heapq.heappop(h)
        if get_ids:
            top_k_ids.append(best_so_far[1])
        
        top_k_result.append(documents[best_so_far[1]])

    return top_k_result if not get_ids else dict(zip(top_k_ids, top_k_result))


def index_exists(paths: Dict[str, str]) -> bool:
    return os.path.isfile(paths['index'])


def load_index(paths: Dict[str, str]):
    global index, doc_lengths, documents

    with open(paths['index'], 'rb') as fp:
        index = pickle.load(fp)
    
    with open(paths['lengths'], 'rb') as fp:
        doc_lengths = pickle.load(fp)

    with open(paths['docs'], 'rb') as fp:
        documents = pickle.load(fp)