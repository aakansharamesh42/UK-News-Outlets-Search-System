from fastapi import APIRouter, Depends, Query
from fastapi.responses import ORJSONResponse
from os.path import basename
from os import getenv
from typing import Optional, Annotated
from pydantic import BaseModel, Field
from utils.basetype import Result
from utils.query_engine import boolean_test, ranked_test, check_query
from utils.redis_utils import (
    caching_query_result,
    get_cache,
    get_docs_fields,
    check_cache_exists,
)
from utils.basetype import RedisKeys, RedisDocKeys
#from ai.QE_Bert import expand_query
from utils.roberta import expand_query
from math import ceil
from utils.spell_checker import SpellChecker
from utils.query_suggestion import QuerySuggestion
from urllib.parse import unquote
from typing import List, Dict, Tuple
from utils.query_expander import QueryExpander
from utils.constant import (
    MONOGRAM_PKL_PATH,
    STOP_WORDS_FILE_PATH,
    FULL_TXT_CORPUS_PATH,
    MONOGRAM_AND_BIGRAM_DICTIONARY_PATH,
)

router = APIRouter(
    prefix=f"/{basename(__file__).replace('.py', '')}",
    tags=[basename(__file__).replace(".py", "")],
    dependencies=[],
    responses={404: {"description": "Not found"}}
)


class SearchResponse(BaseModel):
    results: list[Result]
    truth_value: float


@router.get("/")
async def search(
    q: str = Query(..., description="Search query", min_length=1, max_length=1024),
    year: Optional[int] = Query(
        None, description="Year of the result", ge=1900, le=2100
    ),
    page: Optional[int] = Query(1, description="Page number", ge=1),
    limit: Optional[int] = Query(10, description="Results per page", ge=1, le=100)
    ) -> SearchResponse:
    r'''
    Searching the results from the database.
    ```
        - q: query to search
        - page: page number
        - limit: results per page
    ```
    '''
    ## Search the results
    return ORJSONResponse(content={"results": ['123213'], "truth_value": 0.0})

class TestBody(BaseModel):
    field: str = Field(..., description="Test field", min_length=1, max_length=1024)


@router.post("/test2")
async def test(body: TestBody):
    test_env = getenv("TESTING", "default")
    return ORJSONResponse(content={"field": body.field, "env": test_env})

def paginate_doc_ids(doc_ids: List[int], current_page: int, limit: int, total_pages: int) -> Dict[int, List[int]]:
    """Function to paginated doc_ids"""
    start_page = max(current_page - 4, 1)
    end_page = min(current_page + 4, total_pages)
    page_doc_ids_dict = {}
    for page in range(start_page, end_page + 1):
        page_doc_ids_dict[page] = doc_ids[(page - 1) * limit : page * limit]
    return page_doc_ids_dict

@router.get("/boolean")
async def boolean_search(
    q: str = Query(..., description="Search query", min_length=1, max_length=1024),
    page: Optional[int] = Query(1, description="Page number", ge=1),
    limit: Optional[int] = Query(10, description="Results per page", ge=1, le=100),
):
    r"""
    Searching the results from the database.
    ```
        - q: query to search (Must be a boolean query with AND, OR, NOT, brackets, proximity, exact match and word match)
        - page: page number (default: 1)
        - limit: results per page (default: 10)
    ```
    """

    q = unquote(q)
    
    # uncomment this when the caching is ready
    if await check_cache_exists(RedisKeys.cache("boolean", q, page)):
        results = await get_cache(RedisKeys.cache("boolean", q, page))
        return ORJSONResponse(content=results)

    results = await boolean_test([q])
    total_pages = ceil(len(results[0]) / limit)
    if not results or len(results) > page * limit or total_pages == 0:
        return []
    
    
    page_doc_ids_dict = paginate_doc_ids(results[0], page, limit, total_pages)
    page_results = {}
    page_results["results"] = await get_docs_fields(page_doc_ids_dict[page],
                                                        [RedisDocKeys.title,
                                                        RedisDocKeys.topic,
                                                        RedisDocKeys.url,
                                                        RedisDocKeys.source, 
                                                        RedisDocKeys.date, 
                                                        RedisDocKeys.sentiment, 
                                                        RedisDocKeys.summary])
    page_results["total_pages"] = ceil(len(results[0]) / limit)

    await caching_query_result("boolean", q, page_doc_ids_dict, total_pages=total_pages)

    return ORJSONResponse(content=page_results)

def paginate_doc_ids_and_score(results: List[Tuple[int, float]], current_page: int, limit: int, total_pages: int) -> Tuple[Dict[int, List[int]], Dict[int, List[float]]]:
    """Function to paginated doc_ids"""
    start_page = max(current_page - 4, 1)
    end_page = min(current_page + 4, total_pages)
    page_doc_ids_dict = {}
    page_score_dict = {}
    
    for page in range(start_page, end_page + 1):
        page_doc_ids_dict[page] = [t[0] for t in results[(page - 1) * limit : page * limit]]
        page_score_dict[page] = [t[1] for t in results[(page - 1) * limit : page * limit]]
        
    return page_doc_ids_dict, page_score_dict

@router.get("/tfidf")
async def tfidf_search(
    q: str = Query(..., description="Search query", min_length=1, max_length=1024),
    page: Optional[int] = Query(1, description="Page number", ge=1),
    limit: Optional[int] = Query(10, description="Results per page", ge=1, le=100),
):
    r"""
    Searching the results from the database.
    ```
        - q: query to search (Treat every word as a seperated term)
        - page: page number
        - limit: results per page
    ```
    """

    q = unquote(q)
    
    if await check_cache_exists(RedisKeys.cache("tfidf", q, page)):
        results = await get_cache(RedisKeys.cache("tfidf", q, page))
        return ORJSONResponse(content=results)

    results = await ranked_test([q])
    total_pages = ceil(len(results[0]) / limit)
    if not results or len(results) > page * limit or total_pages == 0:
        return []

    page_doc_ids_dict, page_score_dict = paginate_doc_ids_and_score(results[0], page, limit, total_pages)
    
    page_results = {}
    page_results["results"] = await get_docs_fields(page_doc_ids_dict[page],
                                                        [RedisDocKeys.title,
                                                        RedisDocKeys.topic,
                                                        RedisDocKeys.url, 
                                                        RedisDocKeys.source, 
                                                        RedisDocKeys.date, 
                                                        RedisDocKeys.sentiment, 
                                                        RedisDocKeys.summary])
    page_results["total_pages"] = total_pages
    page_results["scores"] = page_score_dict[page]
    
    await caching_query_result("tfidf", q, page_doc_ids_dict, total_pages=total_pages, scores=page_score_dict)
    
    return ORJSONResponse(content=page_results)


spell_checker = SpellChecker(dictionary_path=MONOGRAM_PKL_PATH)


@router.get("/spellcheck")
async def spellcheck(
    q: str = Query(..., description="Search query", min_length=1, max_length=1024)
):
    r"""
    Spell checking the query string. Returns a corrected string.
    ```
        - q: query to search (Treat every word as a seperated term). must be a string.
    ```
    """
    # spell_checker.correct_query("bidan vs trumpp uneted stetes of amurica"))
    q=unquote(q)
    return spell_checker.correct_query(q)

@router.get("/validate-boolean-query")
async def validate_boolean_query(
    q: str = Query(..., description="Search query", min_length=1, max_length=1024)
):
    r"""
    Spell checking the query string. Returns a corrected string.
    ```
        - q: query to search (Treat every word as a seperated term). must be a string.
    ```
    """
    # spell_checker.correct_query("bidan vs trumpp uneted stetes of amurica"))
    q=unquote(q)
    return check_query(q)


# query_suggestion = QuerySuggestion(monogram_pkl_path=MONOGRAM_PKL_PATH)
# query_suggestion.load_words(words_path=MONOGRAM_AND_BIGRAM_DICTIONARY_PATH)

# @router.post("/suggest_query")
# async def suggestquery(
#     q: str = Query(
#         ..., description="Search query", min_length=1, max_length=1024, size=5
#     )
# ):
#     r"""
#     Query suggestion for the query string. Returns a list of suggested strings.
#     ```
#         - q: query to search (Treat every word as a seperated term)
#     ```
#     """
#     # spell_checker.correct_query("bidan vs trumpp uneted stetes of amurica"))
#     suggestions =  query_suggestion.get_query_suggestions(q)
#     return ORJSONResponse(content={"expanded_queries": suggestions})


# Query with bigram model
# @router.post("/expand-query/")
# async def expand_query_api(query_data: ExpansionQuery):
    

#     # expanded_query = expand_query(query_data.query, query_data.num_expansions)
#     suggestions =  query_suggestion.get_query_suggestions(query_data.query)
#     # return ORJSONResponse(content={"expanded_queries": expanded_query})
#     return ORJSONResponse(content={"expanded_queries": suggestions})

# Query expansion with word2vec
expander = QueryExpander()
@router.get("/query-expansion")
async def query_expansion(
    q: str = Query(..., description="Search query", min_length=1, max_length=1024),
    expansions: int = Query(3, description="Number of expansions", ge=1, le=10)
):
    expanded_query, added_terms = expander.expand_query(q, expansions)
    return ORJSONResponse(content={"expanded_query": expanded_query, "added_terms": added_terms})

# Query Expansion with Roberta
class ExpansionQuery(BaseModel):  
    query: str
    num_expansions: int = 10  # Default value set to 10

# @router.post("/expand-query/")
# async def expand_query_api(query_data: ExpansionQuery):
    

#     expanded_query = expand_query(query_data.query, query_data.num_expansions)
#     return ORJSONResponse(content={"expanded_queries": expanded_query})


# query suggestion with bigram bk trees
query_suggestion = QuerySuggestion(monogram_pkl_path=MONOGRAM_PKL_PATH)
query_suggestion.load_words(words_path=MONOGRAM_AND_BIGRAM_DICTIONARY_PATH)

# this is not query expansion, but SUGGESTION, but leaving it like this in order 
# to not break the frontend
@router.post("/expand-query/")
async def suggest_query_api(query_data: ExpansionQuery):
    """
    Query suggestion for the query string. Returns a list of suggested strings.
    ```
        - q: query to search (string)
    ```
    """
    suggestions =  query_suggestion.get_query_suggestions(query_data.query)
    return ORJSONResponse(content={"expanded_queries": suggestions})
