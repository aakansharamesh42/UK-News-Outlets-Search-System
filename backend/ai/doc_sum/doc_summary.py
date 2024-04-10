import csv
from typing import List
from search_engine import Article
import search_engine as engine
import sys
import nltk
import re
import numpy as np
import networkx as nx
import os
from nltk.corpus import stopwords

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
# print the current working directory
print("Current working directory: ", os.getcwd())

#Article is a class in search engine which has two fields title and body.

maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

print("Max field size limit: ", maxInt)
# nltk.download('stopwords')
stop_words = stopwords.words('english')

#load document paths
def read_data(path: str = "articles50000.csv") -> List[Article]:
    result = {}
    with open(path, 'r', encoding='utf-8') as csvfile:
        articles = csv.reader(csvfile, delimiter=',')
        next(articles, None)
        for aid, article in enumerate(articles):
            result[aid] = Article(title=article[1], body=article[2])
    
    return result


def clean_text(text: str) -> str:
   
    clean_text = re.sub(r'[’”“]', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text


def get_text_sentences(text: str) -> List[str]:
    
    new_text = clean_text(text)
    sentences = nltk.sent_tokenize(new_text)
    return sentences


def preprocess(text: str, remove_stop: bool=True) -> str:
    
    return [t for t in engine.preprocess(text) if t not in stop_words]


def get_similarity(sent1: str, sent2: str) -> np.float32:
    
    terms1 = preprocess(sent1)
    terms2 = preprocess(sent2)

    terms = list(set(terms1 + terms2))

    sent1_vec = np.zeros(len(terms), dtype='int32')
    sent2_vec = np.zeros(len(terms), dtype='int32')

    for term in terms1:
        sent1_vec[terms.index(term)] += 1
    
    for term in terms2:
        sent2_vec[terms.index(term)] += 1
    
    return 1 - nltk.cluster.cosine_distance(sent1_vec, sent2_vec)


def get_similarity_matrix(sentences: List[str]) -> np.ndarray:
  
    n = len(sentences)
    sim_matrix = np.zeros((n, n), dtype='float32')

    for i in range(n):
        for j in range(n):
            if i != j:
                sim_matrix[i, j] = get_similarity(sentences[i], sentences[j])
        
    return sim_matrix


def cosine_pagerank(doc: Article, query: str, sentence_cnt: int) -> str:
    
    result = [doc.title, '\n']
    
    sentences = get_text_sentences(doc.body)
    similarity_matrix = get_similarity_matrix(sentences)
    
    graph = nx.from_numpy_matrix(similarity_matrix)
    sentence_scores = nx.pagerank(graph)

    sentence_scores = sorted(sentence_scores.items(), key=lambda kv: kv[1], reverse=True)
    for i in range(sentence_cnt):
        result.append(sentences[sentence_scores[i][0]] + ' ')
    
    return ''.join(result)


def doc_sum(doc: Article, query: str, sentence_cnt: int = 5):
    
    if len(get_text_sentences(doc.body)) < sentence_cnt:
        raise ValueError("Retrieved article has" +
            f"less than {sentence_cnt} sentences")
    
    print(f"\nDocument summary for Cosine Pagerank Algorithm\n") 
    #print(f"Title: {doc.title}\n")
    print(cosine_pagerank(doc, query, sentence_cnt))


def launch():
    data_path = os.path.join(os.path.dirname(__file__), 'article100.csv')
    save_dir = os.path.join(os.path.dirname(__file__), 'save/')
    save_paths = {
        'index': f'{save_dir}index.p',
        'lengths': f'{save_dir}doc_lengths.p',
        'docs': f'{save_dir}documents.p'
    }
    
    if not engine.index_exists(paths=save_paths):
        print("* Building index... *")
        articles = read_data(data_path)
        engine.build_index(docs=articles, paths=save_paths, dir=save_dir)
        print("* Index was built successfully! *")
    else:
        print("* Loading index... *")
        engine.load_index(paths=save_paths)
        print("* Index was loaded successfully! *")
    
    query = "Trump and tax"
    #replace answer_query with the method in search engine which will fetch and return the top k documents and engine with the filename. [Note search engine is imported as engine]
    docs = engine.answer_query(query, 2)
    print(docs[0])
    #param - 3 represents the num of lines for the output summary.
    doc_sum(docs[0], query, 3)


if __name__ == '__main__':
    launch()