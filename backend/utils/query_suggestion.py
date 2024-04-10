from threading import Lock
import io
import os
import csv
import sys
import string
from collections import defaultdict, deque
from itertools import islice
from enum import Enum
from threading import Lock
from symspellpy import SymSpell
import re
from collections import Counter
import orjson
from typing import List

from constant import (
    STOP_WORDS_FILE_PATH,
    MONOGRAM_PKL_PATH,
    MONOGRAM_AND_BIGRAM_DICTIONARY_PATH,
    FULL_TXT_CORPUS_PATH,
)

try:
    import termios
    import fcntl
except Exception:
    termios = fcntl = None

NORMALIZED_CACHE_SIZE = 2048
MAX_WORD_LENGTH = 40

with open(STOP_WORDS_FILE_PATH, "r") as file:
    FULL_STOP_WORDS = file.read().split("\n")


class CacheNode:
    """
    LFU cache Written by Shane Wang, modified by Sep Dehpour
    """

    def __init__(self, key, value, freq_node, pre, nxt):
        self.key = key
        self.value = value
        self.freq_node = freq_node
        self.pre = pre  # previous CacheNode
        self.nxt = nxt  # next CacheNode

    def free_myself(self):
        if self.freq_node.cache_head == self.freq_node.cache_tail:
            self.freq_node.cache_head = self.freq_node.cache_tail = None
        elif self.freq_node.cache_head == self:
            self.nxt.pre = None
            self.freq_node.cache_head = self.nxt
        elif self.freq_node.cache_tail == self:
            self.pre.nxt = None
            self.freq_node.cache_tail = self.pre
        else:
            self.pre.nxt = self.nxt
            self.nxt.pre = self.pre

        self.pre = None
        self.nxt = None
        self.freq_node = None


class FreqNode:
    def __init__(self, freq, pre, nxt):
        self.freq = freq
        self.pre = pre  # previous FreqNode
        self.nxt = nxt  # next FreqNode
        self.cache_head = None  # CacheNode head under this FreqNode
        self.cache_tail = None  # CacheNode tail under this FreqNode

    def count_caches(self):
        if self.cache_head is None and self.cache_tail is None:
            return 0
        elif self.cache_head == self.cache_tail:
            return 1
        else:
            return "2+"

    def remove(self):
        if self.pre is not None:
            self.pre.nxt = self.nxt
        if self.nxt is not None:
            self.nxt.pre = self.pre

        pre = self.pre
        nxt = self.nxt
        self.pre = self.nxt = self.cache_head = self.cache_tail = None

        return (pre, nxt)

    def pop_head_cache(self):
        if self.cache_head is None and self.cache_tail is None:
            return None
        elif self.cache_head == self.cache_tail:
            cache_head = self.cache_head
            self.cache_head = self.cache_tail = None
            return cache_head
        else:
            cache_head = self.cache_head
            self.cache_head.nxt.pre = None
            self.cache_head = self.cache_head.nxt
            return cache_head

    def append_cache_to_tail(self, cache_node):
        cache_node.freq_node = self

        if self.cache_head is None and self.cache_tail is None:
            self.cache_head = self.cache_tail = cache_node
        else:
            cache_node.pre = self.cache_tail
            cache_node.nxt = None
            self.cache_tail.nxt = cache_node
            self.cache_tail = cache_node

    def insert_after_me(self, freq_node):
        freq_node.pre = self
        freq_node.nxt = self.nxt

        if self.nxt is not None:
            self.nxt.pre = freq_node

        self.nxt = freq_node

    def insert_before_me(self, freq_node):
        if self.pre is not None:
            self.pre.nxt = freq_node

        freq_node.pre = self.pre
        freq_node.nxt = self
        self.pre = freq_node


class LFUCache:

    def __init__(self, capacity):
        self.cache = {}  # {key: cache_node}
        self.capacity = capacity
        self.freq_link_head = None
        self.lock = Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                cache_node = self.cache[key]
                freq_node = cache_node.freq_node
                value = cache_node.value

                self.move_forward(cache_node, freq_node)

                return value
            else:
                return -1

    def set(self, key, value):
        with self.lock:
            if self.capacity <= 0:
                return -1

            if key not in self.cache:
                if len(self.cache) >= self.capacity:
                    self.dump_cache()

                self.create_cache_node(key, value)
            else:
                cache_node = self.cache[key]
                freq_node = cache_node.freq_node
                cache_node.value = value

                self.move_forward(cache_node, freq_node)

    def move_forward(self, cache_node, freq_node):
        if freq_node.nxt is None or freq_node.nxt.freq != freq_node.freq + 1:
            target_freq_node = FreqNode(freq_node.freq + 1, None, None)
            target_empty = True
        else:
            target_freq_node = freq_node.nxt
            target_empty = False

        cache_node.free_myself()
        target_freq_node.append_cache_to_tail(cache_node)

        if target_empty:
            freq_node.insert_after_me(target_freq_node)

        if freq_node.count_caches() == 0:
            if self.freq_link_head == freq_node:
                self.freq_link_head = target_freq_node

            freq_node.remove()

    def dump_cache(self):
        head_freq_node = self.freq_link_head
        self.cache.pop(head_freq_node.cache_head.key)
        head_freq_node.pop_head_cache()

        if head_freq_node.count_caches() == 0:
            self.freq_link_head = head_freq_node.nxt
            head_freq_node.remove()

    def create_cache_node(self, key, value):
        cache_node = CacheNode(key, value, None, None, None)
        self.cache[key] = cache_node

        if self.freq_link_head is None or self.freq_link_head.freq != 0:
            new_freq_node = FreqNode(0, None, None)
            new_freq_node.append_cache_to_tail(cache_node)

            if self.freq_link_head is not None:
                self.freq_link_head.insert_before_me(new_freq_node)

            self.freq_link_head = new_freq_node
        else:
            self.freq_link_head.append_cache_to_tail(cache_node)

    def get_sorted_cache_keys(self):
        result = [(i, freq.freq_node.freq) for i, freq in self.cache.items()]
        result.sort(key=lambda x: -x[1])
        return result


_normalized_lfu_cache = LFUCache(NORMALIZED_CACHE_SIZE)


class FileNotFound(ValueError):
    pass


def _check_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFound(f"{path} does not exist")


def read_csv_gen(path_or_stringio, csv_func=csv.reader, **kwargs):
    """
    Takes a path_or_stringio to a file or a StringIO object and creates a CSV generator
    """
    if isinstance(path_or_stringio, (str, bytes)):
        _check_file_exists(path_or_stringio)
        encoding = kwargs.pop("encoding", "utf-8-sig")
        with open(path_or_stringio, "r", encoding=encoding) as csvfile:
            for i in csv_func(csvfile, **kwargs):
                yield i
    elif isinstance(path_or_stringio, io.StringIO):
        for i in csv_func(path_or_stringio, **kwargs):
            yield i
    else:
        raise TypeError(
            "Either a path to the file or StringIO object needs to be passed."
        )


def _extend_and_repeat(list1, list2):
    if not list1:
        return [[i] for i in list2]

    result = []
    for item in list2:
        if item not in list1:
            list1_copy = list1.copy()
            if item.startswith(list1_copy[-1]):
                list1_copy.pop()
            list1_copy.append(item)
            result.append(list1_copy)

    return result


def read_single_keypress():
    """Waits for a single keypress on stdin.
    https://stackoverflow.com/a/6599441/1497443

    This is a silly function to call if you need to do it a lot because it has
    to store stdin's current setup, setup stdin for reading single keystrokes
    then read the single keystroke then revert stdin back after reading the
    keystroke.

    Returns the character of the key that was pressed (zero on
    KeyboardInterrupt which can happen when a signal gets handled)

    """
    if fcntl is None or termios is None:
        raise ValueError(
            "termios and/or fcntl packages are not available in your system. This is possible because you are not on a Linux Distro."
        )
    fd = sys.stdin.fileno()
    # save old state
    flags_save = fcntl.fcntl(fd, fcntl.F_GETFL)
    attrs_save = termios.tcgetattr(fd)
    # make raw - the way to do this comes from the termios(3) man page.
    attrs = list(attrs_save)  # copy the stored version to update
    # iflag
    attrs[0] &= ~(
        termios.IGNBRK
        | termios.BRKINT
        | termios.PARMRK
        | termios.ISTRIP
        | termios.INLCR
        | termios.IGNCR
        | termios.ICRNL
        | termios.IXON
    )
    # oflag
    attrs[1] &= ~termios.OPOST
    # cflag
    attrs[2] &= ~(termios.CSIZE | termios.PARENB)
    attrs[2] |= termios.CS8
    # lflag
    attrs[3] &= ~(
        termios.ECHONL | termios.ECHO | termios.ICANON | termios.ISIG | termios.IEXTEN
    )
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    # turn off non-blocking
    fcntl.fcntl(fd, fcntl.F_SETFL, flags_save & ~os.O_NONBLOCK)
    # read a single keystroke
    try:
        ret = sys.stdin.read(1)  # returns a single character
    except KeyboardInterrupt:
        ret = 0
    finally:
        # restore old state
        termios.tcsetattr(fd, termios.TCSAFLUSH, attrs_save)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags_save)
    return ret


class Normalizer:

    def __init__(
        self,
        valid_chars_for_string=None,
        valid_chars_for_integer=None,
        valid_chars_for_node_name=None,
    ):
        if valid_chars_for_string:
            self.valid_chars_for_string = frozenset(valid_chars_for_string)
        else:
            self.valid_chars_for_string = frozenset(
                {i for i in string.ascii_letters.lower()}
            )
        if valid_chars_for_integer:
            self.valid_chars_for_integer = frozenset(valid_chars_for_integer)
        else:
            self.valid_chars_for_integer = frozenset({i for i in string.digits})
        if valid_chars_for_node_name:
            self.valid_chars_for_node_name = valid_chars_for_node_name
        else:
            self.valid_chars_for_node_name = self._get_valid_chars_for_node_name()

    def _get_valid_chars_for_node_name(self):
        return (
            {" ", "-", ":", "_"}
            | self.valid_chars_for_string
            | self.valid_chars_for_integer
        )

    def normalize_node_name(self, name, extra_chars=None):
        if name is None:
            return ""
        name = name[:MAX_WORD_LENGTH]
        key = name if extra_chars is None else f"{name}{extra_chars}"
        result = _normalized_lfu_cache.get(key)
        if result == -1:
            result = self._get_normalized_node_name(name, extra_chars=extra_chars)
            _normalized_lfu_cache.set(key, result)
        return result

    def _remove_invalid_chars(self, x):
        result = x in self.valid_chars_for_node_name
        if x == "-" == self.prev_x:
            result = False
        self.prev_x = x
        return result

    def remove_any_special_character(self, name):
        """
        Only remove invalid characters from a name. Useful for cleaning the user's original word.
        """
        if name is None:
            return ""
        name = name.lower()[:MAX_WORD_LENGTH]
        self.prev_x = ""

        return "".join(filter(self._remove_invalid_chars, name)).strip()

    def _get_normalized_node_name(self, name, extra_chars=None):
        name = name.lower()
        result = []
        last_i = None
        for i in name:
            if i in self.valid_chars_for_node_name or (
                extra_chars and i in extra_chars
            ):
                if i == "-":
                    i = " "
                elif (
                    i in self.valid_chars_for_integer
                    and last_i in self.valid_chars_for_string
                ) or (
                    i in self.valid_chars_for_string
                    and last_i in self.valid_chars_for_integer
                ):
                    result.append(" ")
                if not (i == last_i == " "):
                    result.append(i)
                    last_i = i
        return "".join(result).strip()


try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    try:
        from pylev import levenshtein as levenshtein_distance
    except ImportError:
        raise RuntimeError()

DELIMITER = "__"
ORIGINAL_KEY = "original_key"
INF = float("inf")


class NodeNotFound(ValueError):
    pass


class FindStep(Enum):
    start = 0
    descendants_only = 1
    fuzzy_try = 2
    fuzzy_found = 3
    rest_of_fuzzy_round2 = 4
    not_enough_results_add_some_descandants = 5


class _DawgNode:
    """
    The Dawg data structure keeps a set of words, organized with one node for
    each letter. Each node has a branch for each letter that may follow it in the
    set of words.
    """

    __slots__ = ("word", "original_key", "children", "count")

    def __init__(self):
        self.word = None
        self.original_key = None
        self.children = {}
        self.count = 0

    def __getitem__(self, key):
        return self.children[key]

    def __repr__(self):
        return f"<DawgNode children={list(self.children.keys())}, {self.word}>"

    @property
    def value(self):
        return self.original_key or self.word

    def insert(
        self,
        word,
        normalized_word,
        add_word=True,
        original_key=None,
        count=0,
        insert_count=True,
    ):
        node = self
        for letter in normalized_word:
            if letter not in node.children:
                node.children[letter] = _DawgNode()

            node = node.children[letter]

        if add_word:
            node.word = word
            node.original_key = original_key
            if insert_count:
                node.count = int(count)  # converts any str to int
        return node

    def get_descendants_nodes(
        self, size, should_traverse=True, full_stop_words=None, insert_count=True
    ):
        if insert_count is True:
            size = INF

        que = deque()
        unique_nodes = {self}
        found_nodes_set = set()
        full_stop_words = full_stop_words if full_stop_words else set()

        for letter, child_node in self.children.items():
            if child_node not in unique_nodes:
                unique_nodes.add(child_node)
                que.append((letter, child_node))

        while que:
            letter, child_node = que.popleft()
            child_value = child_node.value
            if child_value:
                if child_value in full_stop_words:
                    should_traverse = False
                if child_value not in found_nodes_set:
                    found_nodes_set.add(child_value)
                    yield child_node
                    if len(found_nodes_set) > size:
                        break

            if should_traverse:
                for letter, grand_child_node in child_node.children.items():
                    if grand_child_node not in unique_nodes:
                        unique_nodes.add(grand_child_node)
                        que.append((letter, grand_child_node))

    def get_descendants_words(
        self, size, should_traverse=True, full_stop_words=None, insert_count=True
    ):
        found_nodes_gen = self.get_descendants_nodes(
            size,
            should_traverse=should_traverse,
            full_stop_words=full_stop_words,
            insert_count=insert_count,
        )

        if insert_count is True:
            found_nodes = sorted(
                found_nodes_gen, key=lambda node: node.count, reverse=True
            )[: size + 1]
        else:
            found_nodes = islice(found_nodes_gen, size)

        return map(lambda word: word.value, found_nodes)


def generate_bigrams(text):
    # Tokenize the text by whitespace and punctuation
    words = re.findall(
        r"\b\w+\b", text.lower()
    )  # This regex will match word characters between word boundaries
    # Yield bigrams as tuples
    return zip(words, words[1:])


def calculate_bigram_frequencies(file_path):
    print("Calculating bigram frequencies...")
    bigram_counter = Counter()
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            bigram_counter.update(generate_bigrams(line))
    return bigram_counter


class QuerySuggestion:

    CACHE_SIZE = 2048
    SHOULD_INCLUDE_COUNT = True

    def __init__(
        self,
        monogram_pkl_path: str = MONOGRAM_PKL_PATH,
        synonyms=None,
        full_stop_words: List[str] = FULL_STOP_WORDS,
        logger=None,
        valid_chars_for_string=None,
        valid_chars_for_integer=None,
        valid_chars_for_node_name=None,
    ):
        """
        Initializes the Autocomplete module

        # :param words: A dictionary of words mapped to their context
        :param synonyms: (optional) A dictionary of words to their synonyms.
                         The synonym words should only be here and not repeated in words parameter.
        """
        self._lock = Lock()
        self._dwg = None
        self._raw_synonyms = synonyms or {}
        self._lfu_cache = LFUCache(self.CACHE_SIZE)
        self._clean_synonyms, self._partial_synonyms = (
            self._get_clean_and_partial_synonyms()
        )
        self._reverse_synonyms = self._get_reverse_synonyms(self._clean_synonyms)
        self._full_stop_words = set(full_stop_words) if full_stop_words else None
        self.logger = logger

        spell_checker = SymSpell()
        spell_checker.load_pickle(monogram_pkl_path)
        symspell_words = spell_checker.words
        self.words = {
            key: {"count": int(value / 10)}
            for key, value in symspell_words.items()
            if len(key) > 3 and value > 7 and key not in self._full_stop_words
        }

        self.normalizer = Normalizer(
            valid_chars_for_string=valid_chars_for_string,
            valid_chars_for_integer=valid_chars_for_integer,
            valid_chars_for_node_name=valid_chars_for_node_name,
        )
        new_words = self._get_partial_synonyms_to_words()
        self.words.update(new_words)
        self._populate_dwg()

    def create_monogram_and_bigram_dictionary(
        self,
        full_corpus_txt_path: str = FULL_TXT_CORPUS_PATH,
        output_path: str = MONOGRAM_AND_BIGRAM_DICTIONARY_PATH,
    ):
        monograms = self.words

        print("Creating bigrams...")
        bigram_frequencies = calculate_bigram_frequencies(full_corpus_txt_path)

        # monogram_and_bigram_dictionary = {
        #     monogram: {"count": info["count"]}
        #     for monogram, info in monograms.items()
        #     if "'" not in monogram and monogram not in self._full_stop_words
        # }

        monogram_and_bigram_dictionary = {}
        for monogram, info in monograms.items():
            if (
                "'" not in monogram
                and "’" not in monogram
                and '"' not in monogram
                and monogram not in self._full_stop_words
            ):
                monogram_and_bigram_dictionary[monogram] = {"count": info["count"]}

        print("Creating monogram and bigram dictionary...")
        for bigram, freq in bigram_frequencies.items():
            if (
                freq > 1
                and bigram[0] not in self._full_stop_words
                and bigram[1] not in self._full_stop_words
                and bigram[0] in monograms.keys()
                and bigram[1] in monograms.keys()
                and "'" not in bigram[0]
                and "'" not in bigram[1]
                and "’" not in bigram[0]
                and "’" not in bigram[1]
                and '"' not in bigram[0]
                and '"' not in bigram[1]
            ):
                bigram_text = " ".join(
                    bigram
                )  # Join the bigram tuple into a single string
                monogram_and_bigram_dictionary[bigram_text] = {"count": freq}

        with open(
            output_path,
            "wb",
        ) as file:
            file.write(orjson.dumps(monogram_and_bigram_dictionary))
        print(
            "Monogram and bigram dictionary created and saved in ",
            output_path,
        )

    def load_words(self, words_path: str = MONOGRAM_AND_BIGRAM_DICTIONARY_PATH):
        with open(words_path, "r", encoding="utf-8") as file:
            self.words = orjson.loads(file.read())

    def _get_clean_and_partial_synonyms(self):
        """
        Synonyms are words that should produce the same results.

        - For example `beemer` and `bmw` should both give you `bmw`.
        - `alfa` and `alfa romeo` should both give you `alfa romeo`

        The synonyms get divided into 2 groups:

        1. clean synonyms: The 2 words share little or no words. For example `beemer` vs. `bmw`.
        2. partial synonyms: One of the 2 words is a substring of the other one. For example `alfa` and `alfa romeo` or `gm` vs. `gmc`.

        """
        clean_synonyms = {}
        partial_synonyms = {}

        for key, synonyms in self._raw_synonyms.items():
            key = key.strip().lower()
            _clean = []
            _partial = []
            for syn in synonyms:
                syn = syn.strip().lower()
                if key.startswith(syn):
                    _partial.append(syn)
                else:
                    _clean.append(syn)
            if _clean:
                clean_synonyms[key] = _clean
            if _partial:
                partial_synonyms[key] = _partial

        return clean_synonyms, partial_synonyms

    def _get_reverse_synonyms(self, synonyms):
        result = {}
        if synonyms:
            for key, value in synonyms.items():
                for item in value:
                    result[item] = key
        return result

    def _get_partial_synonyms_to_words(self):
        new_words = {}
        for key, value in self.words.items():
            # data is mutable so we copy
            try:
                value = value.copy()
            # data must be named tuple
            except Exception:
                new_value = value._asdict()
                new_value[ORIGINAL_KEY] = key
                value = type(value)(**new_value)
            else:
                value[ORIGINAL_KEY] = key
            for syn_key, syns in self._partial_synonyms.items():
                if key.startswith(syn_key):
                    for syn in syns:
                        new_key = key.replace(syn_key, syn)
                        new_words[new_key] = value
        return new_words

    def _populate_dwg(self):
        if not self._dwg:
            with self._lock:
                if not self._dwg:
                    self._dwg = _DawgNode()
                    for word, value in self.words.items():
                        original_key = value.get(ORIGINAL_KEY)
                        # word = word.strip().lower()
                        count = value.get("count", 0)
                        leaf_node = self.insert_word_branch(
                            word, original_key=original_key, count=count
                        )
                        if leaf_node and self._clean_synonyms:
                            for synonym in self._clean_synonyms.get(word, []):
                                self.insert_word_branch(
                                    synonym,
                                    leaf_node=leaf_node,
                                    add_word=False,
                                    count=count,
                                )

    def insert_word_callback(self, word):
        """
        Once word is inserted, run this.
        """
        pass

    def insert_word_branch(
        self, word, leaf_node=None, add_word=True, original_key=None, count=0
    ):
        """
        Inserts a word into the Dawg.

        :param word: The word to be inserted as a branch of dwg
        :param leaf_node: (optional) The leaf node for the node to merge into in the dwg.
        :param add_word: (Boolean, default: True) Add the word itself at the end of the branch.
                          Usually this is set to False if we are merging into a leaf node and do not
                          want to add the actual word there.
        :param original_key: If the word that is being added was originally another word.
                             For example with synonyms, you might be inserting the word `beemer` but the
                             original key is `bmw`. This parameter might be removed in the future.

        """
        # if word == 'u (2 off)':
        #     import pytest; pytest.set_trace()
        normalized_word = self.normalizer.normalize_node_name(word)
        # sometimes if the word does not have any valid characters, the normalized_word will be empty
        if not normalized_word:
            return
        last_char = normalized_word[-1]

        if leaf_node:
            temp_leaf_node = self._dwg.insert(
                word=word,
                normalized_word=normalized_word[:-1],
                add_word=add_word,
                original_key=original_key,
                count=count,
                insert_count=self.SHOULD_INCLUDE_COUNT,
            )
            # It already has children
            if temp_leaf_node.children and last_char in temp_leaf_node.children:
                temp_leaf_node.children[last_char].word = leaf_node.word
            # otherwise merge into the leaf node
            else:
                temp_leaf_node.children[last_char] = leaf_node
        else:
            leaf_node = self._dwg.insert(
                word=word,
                normalized_word=normalized_word,
                original_key=original_key,
                count=count,
                insert_count=self.SHOULD_INCLUDE_COUNT,
            )
        self.insert_word_callback(word)
        return leaf_node

    def _find_and_sort(self, word, max_cost, size):
        output_keys_set = set()
        results, find_steps = self._find(word, max_cost, size)
        results_keys = list(results.keys())
        results_keys.sort()
        for key in results_keys:
            for output_items in results[key]:
                for i, item in enumerate(output_items):
                    reversed_item = self._reverse_synonyms.get(item)
                    if reversed_item:
                        output_items[i] = reversed_item
                    elif item not in self.words:
                        output_items[i] = item
                output_items_str = DELIMITER.join(output_items)
                if output_items and output_items_str not in output_keys_set:
                    output_keys_set.add(output_items_str)
                    yield output_items
                    if len(output_keys_set) >= size:
                        return

    def get_tokens_flat_list(self, word, max_cost=3, size=10):
        """
        Gets a flat list of tokens.
        This requires the original search function from this class to be run,
        instead of subclasses of AutoComplete.
        """
        result = QuerySuggestion.search(self, word, max_cost=max_cost, size=size)
        return [item for sublist in result for item in sublist]

    def get_word_context(self, word):
        """
        Gets the word's context from the words dictionary
        """
        word = self.normalizer.normalize_node_name(word)
        return self.words.get(word)

    def search(self, word, max_cost=2, size=5):
        """
        parameters:
        - word: the word to return autocomplete results for
        - max_cost: Maximum Levenshtein edit distance to be considered when calculating results
        - size: The max number of results to return
        """
        word = self.normalizer.normalize_node_name(word)
        if not word:
            return []
        key = f"{word}-{max_cost}-{size}"
        result = self._lfu_cache.get(key)
        if result == -1:
            result = list(self._find_and_sort(word, max_cost, size))
            self._lfu_cache.set(key, result)
        # return self.filter_single_word_results(result)
        return result

    def filter_single_word_results(self, original_list):
        filtered_list = []
        appeared_strings = set()

        for sublist in original_list:
            for string in sublist:
                if string not in appeared_strings:
                    filtered_list.append(string)
                    appeared_strings.add(string)
                    break
        return filtered_list

    # def suggest_last_

    @staticmethod
    def _len_results(results):
        return sum(map(len, results.values()))

    @staticmethod
    def _is_enough_results(results, size):
        return QuerySuggestion._len_results(results) >= size

    def _is_stop_word_condition(self, matched_words, matched_prefix_of_last_word):
        return (
            self._full_stop_words
            and matched_words
            and matched_words[-1] in self._full_stop_words
            and not matched_prefix_of_last_word
        )

    def _find(self, word, max_cost, size, call_count=0):
        """
        The search function returns a list of all words that are less than the given
        maximum distance from the target word
        """
        results = defaultdict(list)
        fuzzy_matches = defaultdict(list)
        rest_of_results = {}
        fuzzy_matches_len = 0

        fuzzy_min_distance = min_distance = INF
        matched_prefix_of_last_word, rest_of_word, new_node, matched_words = (
            self._prefix_autofill(word=word)
        )

        last_word = matched_prefix_of_last_word + rest_of_word

        if matched_words:
            results[0] = [matched_words.copy()]
            min_distance = 0
            # under certain condition with finding full stop words, do not bother with finding more matches
            if self._is_stop_word_condition(matched_words, matched_prefix_of_last_word):
                find_steps = [FindStep.start]
                return results, find_steps
        if len(rest_of_word) < 3:
            find_steps = [FindStep.descendants_only]
            self._add_descendants_words_to_results(
                node=new_node,
                size=size,
                matched_words=matched_words,
                results=results,
                distance=1,
            )
        else:
            find_steps = [FindStep.fuzzy_try]
            word_chunks = deque(filter(lambda x: x, last_word.split(" ")))
            new_word = word_chunks.popleft()

            # TODO: experiment with the number here
            # 'in los angeles' gets cut into `in los` so it becomes a closer match to `in lodi`
            # but if the number was bigger, we could have matched with `in los angeles`
            while len(new_word) < 5 and word_chunks:
                new_word = f"{new_word} {word_chunks.popleft()}"
            fuzzy_rest_of_word = " ".join(word_chunks)

            for _word in self.words:
                if abs(len(_word) - len(new_word)) > max_cost:
                    continue
                dist = levenshtein_distance(new_word, _word)
                if dist < max_cost:
                    fuzzy_matches_len += 1
                    _value = self.words[_word].get(ORIGINAL_KEY, _word)
                    fuzzy_matches[dist].append(_value)
                    fuzzy_min_distance = min(fuzzy_min_distance, dist)
                    if fuzzy_matches_len >= size or dist < 2:
                        break
            if fuzzy_matches_len:
                find_steps.append(FindStep.fuzzy_found)
                if fuzzy_rest_of_word:
                    call_count += 1
                    if call_count < 2:
                        rest_of_results, rest_find_steps = self._find(
                            word=fuzzy_rest_of_word,
                            max_cost=max_cost,
                            size=size,
                            call_count=call_count,
                        )
                        find_steps.append(
                            {FindStep.rest_of_fuzzy_round2: rest_find_steps}
                        )
                for _word in fuzzy_matches[fuzzy_min_distance]:
                    if rest_of_results:
                        rest_of_results_min_key = min(rest_of_results.keys())
                        for _rest_of_matched_word in rest_of_results[
                            rest_of_results_min_key
                        ]:
                            results[fuzzy_min_distance].append(
                                matched_words + [_word] + _rest_of_matched_word
                            )
                    else:
                        results[fuzzy_min_distance].append(matched_words + [_word])
                        (
                            _matched_prefix_of_last_word_b,
                            not_used_rest_of_word,
                            fuzzy_new_node,
                            _matched_words_b,
                        ) = self._prefix_autofill(word=_word)
                        if self._is_stop_word_condition(
                            matched_words=_matched_words_b,
                            matched_prefix_of_last_word=_matched_prefix_of_last_word_b,
                        ):
                            break
                        self._add_descendants_words_to_results(
                            node=fuzzy_new_node,
                            size=size,
                            matched_words=matched_words,
                            results=results,
                            distance=fuzzy_min_distance,
                        )

            if matched_words and not self._is_enough_results(results, size):
                find_steps.append(FindStep.not_enough_results_add_some_descandants)
                total_min_distance = min(min_distance, fuzzy_min_distance)
                self._add_descendants_words_to_results(
                    node=new_node,
                    size=size,
                    matched_words=matched_words,
                    results=results,
                    distance=total_min_distance + 1,
                )

        return results, find_steps

    def _prefix_autofill(self, word, node=None):
        len_prev_rest_of_last_word = INF
        matched_words = []
        matched_words_set = set()

        def _add_words(words):
            is_added = False
            for word in words:
                if word not in matched_words_set:
                    matched_words.append(word)
                    matched_words_set.add(word)
                    is_added = True
            return is_added

        (
            matched_prefix_of_last_word,
            rest_of_word,
            node,
            matched_words_part,
            matched_condition_ever,
            matched_condition_in_branch,
        ) = self._prefix_autofill_part(word, node)
        _add_words(matched_words_part)
        result = (matched_prefix_of_last_word, rest_of_word, node, matched_words)
        len_rest_of_last_word = len(rest_of_word)

        while (
            len_rest_of_last_word and len_rest_of_last_word < len_prev_rest_of_last_word
        ):
            word = matched_prefix_of_last_word + rest_of_word
            word = word.strip()
            len_prev_rest_of_last_word = len_rest_of_last_word
            (
                matched_prefix_of_last_word,
                rest_of_word,
                node,
                matched_words_part,
                matched_condition_ever,
                matched_condition_in_branch,
            ) = self._prefix_autofill_part(
                word,
                node=self._dwg,
                matched_condition_ever=matched_condition_ever,
                matched_condition_in_branch=matched_condition_in_branch,
            )
            is_added = _add_words(matched_words_part)
            if is_added is False:
                break
            len_rest_of_last_word = len(rest_of_word)
            result = (matched_prefix_of_last_word, rest_of_word, node, matched_words)

        return result

    def prefix_autofill_part_condition(self, node):
        pass

    PREFIX_AUTOFILL_PART_CONDITION_SUFFIX = ""

    def _add_to_matched_words(
        self,
        node,
        matched_words,
        matched_condition_in_branch,
        matched_condition_ever,
        matched_prefix_of_last_word,
    ):
        if matched_words:
            last_matched_word = matched_words[-1].replace(
                self.PREFIX_AUTOFILL_PART_CONDITION_SUFFIX, ""
            )
            if node.value.startswith(last_matched_word):
                matched_words.pop()
        value = node.value
        if self.PREFIX_AUTOFILL_PART_CONDITION_SUFFIX:
            if self._node_word_info_matches_condition(
                node, self.prefix_autofill_part_condition
            ):
                matched_condition_in_branch = True
                if matched_condition_ever and matched_prefix_of_last_word:
                    value = f"{matched_prefix_of_last_word}{self.PREFIX_AUTOFILL_PART_CONDITION_SUFFIX}"
        matched_words.append(value)
        return matched_words, matched_condition_in_branch

    def _prefix_autofill_part(
        self,
        word,
        node=None,
        matched_condition_ever=False,
        matched_condition_in_branch=False,
    ):
        node = node or self._dwg
        que = deque(word)

        matched_prefix_of_last_word = ""
        matched_words = []
        nodes_that_words_were_extracted = set()

        while que:
            char = que.popleft()

            if node.children:
                if char not in node.children:
                    space_child = node.children.get(" ")
                    if space_child and char in space_child.children:
                        node = space_child
                    else:
                        que.appendleft(char)
                        break
                node = node.children[char]
                if char != " " or matched_prefix_of_last_word:
                    matched_prefix_of_last_word += char
                if node.word:
                    if que:
                        next_char = que[0]
                        if next_char != " ":
                            continue
                    matched_words, matched_condition_in_branch = (
                        self._add_to_matched_words(
                            node,
                            matched_words,
                            matched_condition_in_branch,
                            matched_condition_ever,
                            matched_prefix_of_last_word,
                        )
                    )
                    nodes_that_words_were_extracted.add(node)
                    matched_prefix_of_last_word = ""
            else:
                if char == " ":
                    node = self._dwg
                    if matched_condition_in_branch:
                        matched_condition_ever = True
                else:
                    que.appendleft(char)
                    break

        if not que and node.word and node not in nodes_that_words_were_extracted:
            matched_words, matched_condition_in_branch = self._add_to_matched_words(
                node,
                matched_words,
                matched_condition_in_branch,
                matched_condition_ever,
                matched_prefix_of_last_word,
            )
            matched_prefix_of_last_word = ""

        rest_of_word = "".join(que)
        if matched_condition_in_branch:
            matched_condition_ever = True

        return (
            matched_prefix_of_last_word,
            rest_of_word,
            node,
            matched_words,
            matched_condition_ever,
            matched_condition_in_branch,
        )

    def _add_descendants_words_to_results(
        self, node, size, matched_words, results, distance, should_traverse=True
    ):
        descendant_words = list(
            node.get_descendants_words(
                size, should_traverse, full_stop_words=self._full_stop_words
            )
        )
        extended = _extend_and_repeat(matched_words, descendant_words)
        if extended:
            results[distance].extend(extended)
        return distance

    def _node_word_info_matches_condition(self, node, condition):
        _word = node.word
        word_info = self.words.get(_word)
        if word_info:
            return condition(word_info)
        else:
            return False

    def get_all_descendent_words_for_condition(self, word, size, condition):
        """
        This is used in the search tokenizer not in the fast autocomplete itself.
        """
        new_tokens = []

        (
            matched_prefix_of_last_word,
            rest_of_word,
            node,
            matched_words_part,
            matched_condition_ever,
            matched_condition_in_branch,
        ) = self._prefix_autofill_part(word=word)
        if not rest_of_word and self._node_word_info_matches_condition(node, condition):
            found_nodes_gen = node.get_descendants_nodes(
                size, insert_count=self.SHOULD_INCLUDE_COUNT
            )
            for node in found_nodes_gen:
                if self._node_word_info_matches_condition(node, condition):
                    new_tokens.append(node.word)
        return new_tokens

    def update_count_of_word(self, word, count=None, offset=None):
        """
        Update the count attribute of a node in the dwg. This only affects the autocomplete
        object and not the original count of the node in the data that was fed into fast_autocomplete.
        """
        (
            matched_prefix_of_last_word,
            rest_of_word,
            node,
            matched_words_part,
            matched_condition_ever,
            matched_condition_in_branch,
        ) = self._prefix_autofill_part(word=word)
        if node:
            if offset:
                with self._lock:
                    node.count += offset
            elif count:
                with self._lock:
                    node.count = count
        else:
            raise NodeNotFound(f"Unable to find a node for word {word}")
        return node.count

    def get_count_of_word(self, word):
        return self.update_count_of_word(word)
    
    def modify_lists_allow_first_repeat(self, lists):
        """
        Modify the to prevent the second/third element from repeating. The first
        string is allowed to repeat.
        """
        seen = set()  # To keep track of non-first strings that have already appeared
        result = []  # To store the modified lists

        for lst in lists:
            if len(lst) == 1:
                result.append(lst)
                continue
            # Always keep the first element
            new_lst = lst[:1] if lst else []

            # For the second and subsequent elements, keep only if not seen before
            for s in lst[1:]:
                if s not in seen:
                    seen.add(s)  # Mark as seen now
                    new_lst.append(s)  # Add to the current list if not seen

            # Append the modified list to the result, ensuring it has at most two strings
            result.append(new_lst[:2])

        return result

    def get_query_suggestions(self, query: str, max_cost: int = 2, size: int = 5):
        """
        Get suggestions for the query
        """
        query = " ".join(
            [word for word in query.split() if word not in FULL_STOP_WORDS]
        )
        if len(query.split()) > 1:
            # if yes, get the last word
            string_to_suggest_off = " ".join(query.split()[-2:])
            prefix = " ".join(query.split()[:-2])
        else:
            string_to_suggest_off = query
            prefix = ""

        answer = self.search(word=(string_to_suggest_off), size=size)

        if not answer and len(query.split()) > 1:
            # if yes, get the last word
            string_to_suggest_off = query.split()[-1]
            prefix = " ".join(query.split()[:-1])
            # print("\nNo suggestions found for two words. Trying with a word.")
            # print("New prefix", prefix)
            # print("New to suggest off:", string_to_suggest_off, "\n")
            answer = self.search(word=(string_to_suggest_off), size=5)
        
        answer_list = self.modify_lists_allow_first_repeat(answer)
        answers = [" ".join([prefix, " ".join(lst)]) for lst in answer_list]
        return answers


class _DawgNode:
    """
    The Dawg data structure keeps a set of words, organized with one node for
    each letter. Each node has a branch for each letter that may follow it in the
    set of words.
    """

    __slots__ = ("word", "original_key", "children", "count")

    def __init__(self):
        self.word = None
        self.original_key = None
        self.children = {}
        self.count = 0

    def __getitem__(self, key):
        return self.children[key]

    def __repr__(self):
        return f"<DawgNode children={list(self.children.keys())}, {self.word}>"

    @property
    def value(self):
        return self.original_key or self.word

    def insert(
        self,
        word,
        normalized_word,
        add_word=True,
        original_key=None,
        count=0,
        insert_count=True,
    ):
        node = self
        for letter in normalized_word:
            if letter not in node.children:
                node.children[letter] = _DawgNode()

            node = node.children[letter]

        if add_word:
            node.word = word
            node.original_key = original_key
            if insert_count:
                node.count = int(count)  # converts any str to int
        return node

    def get_descendants_nodes(
        self, size, should_traverse=True, full_stop_words=None, insert_count=True
    ):
        if insert_count is True:
            size = INF

        que = deque()
        unique_nodes = {self}
        found_nodes_set = set()
        full_stop_words = full_stop_words if full_stop_words else set()

        for letter, child_node in self.children.items():
            if child_node not in unique_nodes:
                unique_nodes.add(child_node)
                que.append((letter, child_node))

        while que:
            letter, child_node = que.popleft()
            child_value = child_node.value
            if child_value:
                if child_value in full_stop_words:
                    should_traverse = False
                if child_value not in found_nodes_set:
                    found_nodes_set.add(child_value)
                    yield child_node
                    if len(found_nodes_set) > size:
                        break

            if should_traverse:
                for letter, grand_child_node in child_node.children.items():
                    if grand_child_node not in unique_nodes:
                        unique_nodes.add(grand_child_node)
                        que.append((letter, grand_child_node))

    def get_descendants_words(
        self, size, should_traverse=True, full_stop_words=None, insert_count=True
    ):
        found_nodes_gen = self.get_descendants_nodes(
            size,
            should_traverse=should_traverse,
            full_stop_words=full_stop_words,
            insert_count=insert_count,
        )

        if insert_count is True:
            found_nodes = sorted(
                found_nodes_gen, key=lambda node: node.count, reverse=True
            )[: size + 1]
        else:
            found_nodes = islice(found_nodes_gen, size)

        return map(lambda word: word.value, found_nodes)



# if __name__ == "__main__":

#     dictionary_path = "C:/Users/Asus/Desktop/ttds-proj/backend/utils/spell_checking_and_autocomplete_files/symspell_dictionary.pkl"
#     query_suggestion = QuerySuggestion(dictionary_path=dictionary_path)
#     query_suggestion.search("ed")