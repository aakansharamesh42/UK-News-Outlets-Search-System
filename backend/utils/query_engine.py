import re
import traceback
import os
import time
import math
import asyncio
import sys
import heapq
sys.path.append(os.path.dirname(__file__))
from nltk.stem import PorterStemmer
from typing import DefaultDict, Dict, List, Tuple, Set
from common import read_file, get_stop_words, get_preprocessed_words
from redis_utils import (
    get_doc_size,
    get_tfidf_doc_size,
    get_tfs,
    get_doc_ids_list,
    get_json_values,
    is_key_exists,
    get_json_value,
)
from build_index import delta_decode_list
from basetype import RedisKeys
from concurrent.futures import ProcessPoolExecutor

# STOP_WORDS_FILE = "ttds_2023_english_stop_words.txt"

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
NUM_OF_CORES = os.cpu_count()
SPECIAL_PATTERN = {
    "proximity": re.compile(r"#(\d+)\((\w+),\s*(\w+)\)"),
    "exact": re.compile(r"\"[^\"]+\""),
    "spliter": re.compile(
        r"(AND|OR|NOT|#\d+\(\w+,\s*\w+\)|\"[^\"]+\"|\'[^\']+\'|\w+|\(|\))"
    ),
}


def preprocess_match(
    match: re.Match, stopping: bool = True, stemming: bool = True
) -> str:
    word = match.group(0)
    if word in ["AND", "OR", "NOT"]:
        return word
    word = word.lower()

    stopwords = get_stop_words()
    if stopping and word in stopwords:
        return ""

    if stemming:
        stemmer = PorterStemmer()
        word = stemmer.stem(word)

    return word


def load_queries(file_name: str) -> list:
    query_lines = read_file(file_name).split("\n")
    queries = []
    for line in query_lines:
        if line == "":
            continue
        query_id, query_text = line.split(":")
        queries.append(query_text.strip())
    return queries


def handle_binary_operator(operator: str, left: list, right: list) -> list:
    # print("handle binary operator", operator, left, right)
    left = [] if left is None else left
    right = [] if right is None else right
    if operator == "AND":
        print("AND operation")
        return list(set(left) & set(right))
    elif operator == "OR":
        print("OR operation")
        return list(set(left) | set(right))


def handle_not_operator(operand: List[int], doc_ids_list: List[int]) -> List[int]:
    print("NOT operation")
    if operand is None:
        return doc_ids_list
    return list(set(doc_ids_list) - set(operand))


async def get_doc_ids_from_string(string: str) -> List[int]:
    # check if string is a phrase bounded by double quotes
    if await is_key_exists(RedisKeys.index(string)):
        term_index = await get_json_value(RedisKeys.index(string))
        return list(map(int, term_index.keys()))
    else:
        return []


async def get_doc_ids_from_pattern(pattern: str) -> List[int]:
    # pattern is of the form "A B"/"A B C" etc
    # retrieve words from the pattern
    doc_ids = []
    words = re.findall(r"\w+", pattern)
    # check if the word are in consecutive positions
    words_index = await get_json_values([RedisKeys.index(word) for word in words])
    for doc_id in words_index[0]:
        positions = delta_decode_list(words_index[0][doc_id])
        for pos in positions:
            try:
                if (
                    all(
                        [
                            pos + i in words_index[i][doc_id]
                            for i in range(1, len(words))
                        ]
                    )
                    and doc_id not in doc_ids
                ):
                    doc_ids.append(int(doc_id))
            except:
                pass

    return doc_ids


def negate_doc_ids(doc_ids: List[int], doc_ids_list: List[int]) -> List[int]:
    return list(set(doc_ids_list) - set(doc_ids))

async def process_doc_id(doc_id, values, n):
    try:
        positions_for_w1 = delta_decode_list(values[0][str(doc_id)])
        positions_for_w2 = delta_decode_list(values[1][str(doc_id)])
        if any(
            [
                abs(pos1 - pos2) <= int(n)
                for pos1 in positions_for_w1
                for pos2 in positions_for_w2
            ]
        ):
            return int(doc_id)
    except:
        pass

async def evaluate_proximity_pattern(n: int, w1: str, w2: str) -> List[int]:
    # find all the doc_ids for w1 and w2
    doc_ids_for_w1 = await get_doc_ids_from_string(w1)
    # find the doc_ids that satisfy the condition
    doc_ids = []
    values = await get_json_values([RedisKeys.index(w1), RedisKeys.index(w2)])
    tasks = []
    for doc_id in doc_ids_for_w1:
        tasks.append(process_doc_id(doc_id, values, n))
    results = await asyncio.gather(*tasks)
    doc_ids = [result for result in results if result is not None]
    return doc_ids


async def evaluate_subquery(
    subquery: str, special_patterns: Dict[str, re.Pattern]
) -> List[int]:

    proximity_match = re.match(special_patterns["proximity"], subquery)
    exact_match = re.match(special_patterns["exact"], subquery)
    if proximity_match:
        n = proximity_match.group(1)
        w1 = proximity_match.group(2)
        w2 = proximity_match.group(3)
        print("Handle proximity pattern", n, w1, w2)
        return await evaluate_proximity_pattern(n, w1, w2)
    else:
        if exact_match:
            print("handle phrase", subquery[1:-1])
            return await get_doc_ids_from_pattern(subquery[1:-1])
        else:
            print("handle word(s)", subquery)
            return await get_doc_ids_from_string(subquery)


def calculate_tf_idf(
    tokens: List[str],
    doc_id: str,
    docs_size: int,
    word_freq: Dict[str, Dict[str, int]],
    doc_freq: Dict[str, int],
) -> float:
    tf_idf_score = 0
    for token in tokens:
        doc_tf = word_freq[token].get(doc_id, 0)
        if doc_tf == 0:
            continue
        tf = 1 + math.log10(doc_tf)
        idf = math.log10(docs_size / doc_freq[token])
        tf_idf_score += tf * idf
    return tf_idf_score

# convert infix to postfix
def precedence(operator: str) -> int:
    if operator == "NOT":
        return 3
    elif operator == "AND" or operator == "OR":
        return 2
    elif operator == "(" or operator == ")":
        return 1
    else:
        return -1


def associativity(operator: str) -> str:
    if operator == "NOT":
        return "right"
    else:
        return "left"


def is_operator(token: str) -> bool:
    return token in ["AND", "OR", "NOT"]


def infix_to_postfix(query: str, spliter: re.Pattern) -> List[str]:
    tokens = re.findall(spliter, query)
    stack = []
    postfix = []
    for token in tokens:
        if is_operator(token):
            while (
                stack
                and is_operator(stack[-1])
                and (
                    (
                        associativity(token) == "left"
                        and precedence(token) <= precedence(stack[-1])
                    )
                    or (
                        associativity(token) == "right"
                        and precedence(token) < precedence(stack[-1])
                    )
                )
            ):
                postfix.append(stack.pop())
            stack.append(token)
        elif token == "(":
            stack.append(token)
        elif token == ")":
            while stack and stack[-1] != "(":
                postfix.append(stack.pop())
            if stack and stack[-1] == "(":
                stack.pop()
        else:
            postfix.append(token)
    while stack:
        postfix.append(stack.pop())
    return postfix


def is_valid_query(query: str) -> bool:
    # check if the query is valid
    spliter = re.compile(
        r"(AND|OR|NOT|#\d+\(\w+,\s*\w+\)|\"[^\"]+\"|\'[^\']+\'|\w+|\(|\))"
    )
    tokens = re.findall(spliter, query)
    prev_token = None
    parentheses_count = 0
    for token in tokens:
        if token == "(":
            parentheses_count += 1
        elif token == ")":
            parentheses_count -= 1
            if parentheses_count < 0:
                print("Parentheses count is less than 0")
                return False
        elif token == "NOT":
            if prev_token and (not is_operator(prev_token) and prev_token != "("):
                print("Invalid NOT position")
                return False
        elif is_operator(token):
            if prev_token and (prev_token == "(" or is_operator(prev_token)):
                print("Invalid operator position")
                return False
        else:
            # token is an operand
            if prev_token and prev_token == ")":
                print("Invalid operand position")
                return False
        prev_token = token
    if parentheses_count != 0:
        return False
    return True

def check_query(query: str, stopping: bool = True, stemming: bool = True, special_patterns: Dict[str, re.Pattern] = SPECIAL_PATTERN) -> bool:
    query = re.sub(r"(\w+)", lambda x: preprocess_match(x, stopping, stemming), query)
    if not is_valid_query(query):
        return False
    return True

async def evaluate_boolean_query(
    query: str,
    doc_ids_list: List[int],
    stopping: bool = True,
    stemming: bool = True,
    special_patterns: Dict[str, re.Pattern] = SPECIAL_PATTERN,
) -> List:
    # query = " ".join([token.lower() if token not in ["AND", "OR", "NOT"] else token for token in query.split("\w+ ")])
    query = re.sub(r"(\w+)", lambda x: preprocess_match(x, stopping, stemming), query)
    # print(query)
    if not is_valid_query(query):
        print("Invalid query: ", query)
        return []

    postfix = infix_to_postfix(query, special_patterns["spliter"])
    # print("postfix", postfix)

    # evalute the value for the stuff first
    results = []
    tasks = [
        evaluate_subquery(token, special_patterns)
        for token in postfix
        if not is_operator(token)
    ]
    results = await asyncio.gather(*tasks)
    for idx, token in enumerate(postfix):
        if not is_operator(token):
            postfix[idx] = results.pop(0)
    try:
        stack = []
        for token in postfix:
            if is_operator(token):
                if token == "NOT":
                    right = stack.pop()
                    result = handle_not_operator(right, doc_ids_list)
                else:
                    right = stack.pop()
                    left = stack.pop()
                    result = handle_binary_operator(token, left, right)
                stack.append(result)
            else:
                # token is an operand
                stack.append(token)
        return stack.pop()

    except:
        # print the processing error term
        traceback.print_exc()
        exit()

# TODO - show added terms in the interface
async def evaluate_ranked_query(
    query: str,
    docs_size: int,
    stopping: bool = True,
    stemming: bool = True,
) -> List[Tuple[int, float]]:
    words = get_preprocessed_words(query, stopping, stemming)
    doc_ids = set()
    doc_freq = dict()
    tfs = await get_tfs(words)
    
    if not tfs:
        return []
    
    for word, tf in tfs.items():
        doc_ids = doc_ids.union(tf.keys())
        doc_freq[word] = len(tf)
    
    scores = []
    score_results = []
    for doc_id in doc_ids:
        score_results.append(calculate_tf_idf(words, doc_id, docs_size, tfs, doc_freq))
    
    for idx, doc_id in enumerate(doc_ids):
        scores.append((doc_id, score_results[idx]))
    
    # sort by the score and the doc_id
    # scores.sort(key=lambda x: (-x[1], x[0]))
    # scores = sorted(scores, key=lambda x: (-x[1], x[0])

    # sort the scores in chunks using the process pool executor
    # check if the length of the scores is greater than 1000
    # if len(scores) > 1000:
    #     scores = heapq.nlargest(1000, scores, key=lambda x: x[1])
    # else:
    scores = sorted(scores, key=lambda x: (-x[1]))
    
    return scores


async def boolean_test(
    boolean_queries: List[str] = ["\"Comic Relief\" AND (NOT wtf OR #1(Comic, Relief))"],
) -> List[List[int]]:
    doc_ids_list = await get_doc_ids_list()
    results = []
    for query in boolean_queries:
        results.append(await evaluate_boolean_query(query, doc_ids_list))
    return results


async def ranked_test(
    ranked_queries: List[str] = ["Comic Relief"],
) -> List[List[Tuple[int, float]]]:
    doc_size = await get_tfidf_doc_size()
    results = []
    for query in ranked_queries:
        results.append(await evaluate_ranked_query(query, doc_size))
    return results


async def main():
    print(await boolean_test(["#1(united, kingdom)"]))
    # await ranked_test(["Donald Trump and Biden in 2024 USA"])

    #### BENCHMARKING
    # result = await ranked_test()
    # print(result[0][:5])

    # Inference Time in Query 1
    # Baseline: Comic Relief: 0.3459899425506592 346
    # Improvement: Comic Relief: 0.28830504417419434 346

    # Inference Time in Query 2
    # Baseline: Donald Trump and Biden in 2024 USA: 3.79338002204895 1765
    # Improvement: Donald Trump and Biden in 2024 USA: 0.5762429237365723 1765

    # Inference Result Query 1
    # Baseline: Comic Relief: [('219148', 6.715275777697791), ('220261', 6.715275777697791), ('222097', 6.715275777697791), ('312028', 6.715275777697791), ('313037', 6.5266929715344775)]
    # Improvement: Comic Relief: [('219148', 6.715275777697791), ('220261', 6.715275777697791), ('222097', 6.715275777697791), ('312028', 6.715275777697791), ('313037', 6.5266929715344775)]

    # Inference Result Query 2
    # Baseline: Donald Trump and Biden in 2024 USA: [('224817', 21.52595725971649), ('220825', 21.123974002575643), ('222614', 20.60367423116946), ('221260', 19.128669330853047), ('222408', 18.69979137178141)]
    # Improvement: Donald Trump and Biden in 2024 USA: [('224817', 21.52595725971649), ('220825', 21.123974002575643), ('222614', 20.60367423116946), ('221260', 19.128669330853047), ('222408', 18.69979137178141)]


if __name__ == "__main__":
    asyncio.run(main())

    # # ### processing ranked queries
    # ranked_queries = read_ranked_queries("queries.ranked.txt")
    # start = time.time()
    # ranked_results = evaluate_ranked_query(ranked_queries, index)
    # save_ranked_queries_result(ranked_results, '')
    # print("Time taken to process ranked queries", time.time() - start)
