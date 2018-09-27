import copy
import re
from typing import List, Tuple

from nltk.tokenize.nist import NISTTokenizer


class BaseTokenizer:
    def tokenize_into_words(self, src: str) -> List[Tuple[int, int]]:
        raise NotImplemented

    @staticmethod
    def tokenize_into_characters(src: str, bounds_of_words: List[Tuple[int, int]]) -> List[str]:
        characters_list = []
        start_pos = 0
        for bounds_of_cur_word in bounds_of_words:
            if bounds_of_cur_word[0] > start_pos:
                characters_list.append('\n' if '\n' in src[start_pos:bounds_of_cur_word[0]] else ' ')
            characters_list += list(src[bounds_of_cur_word[0]:bounds_of_cur_word[1]])
            start_pos = bounds_of_cur_word[1]
        if start_pos < len(src):
            characters_list.append('\n' if '\n' in src[start_pos:] else ' ')
        return characters_list


class DefaultTokenizer(BaseTokenizer):
    def __init__(self, special_symbols: set=None):
        super().__init__()
        self.special_symbols = special_symbols
        self.tokenizer = NISTTokenizer()
        if (self.special_symbols is not None) and (len(self.special_symbols) > 0):
            re_expr = '(' + '|'.join([re.escape(cur) for cur in self.special_symbols]) + ')'
            self.re_for_special_symbols = re.compile(re_expr)
        else:
            self.re_for_special_symbols = None

    def tokenize_into_words(self, src: str) -> List[Tuple[int, int]]:
        prep = src.strip()
        if len(prep) == 0:
            return []
        if self.re_for_special_symbols is None:
            bounds_of_tokens = self.__tokenize_text(src)
        else:
            bounds_of_subphrases = []
            start_pos = 0
            for search_res in self.re_for_special_symbols.finditer(src):
                if (search_res.start() < 0) or (search_res.end() < 0):
                    break
                bounds_of_subphrases.append(('', (start_pos, search_res.start())))
                cur_symbol = src[search_res.start():search_res.end()]
                bounds_of_subphrases.append(
                    (
                        cur_symbol,
                        (search_res.start(), search_res.end())
                    )
                )
                start_pos = search_res.end()
            if start_pos < len(src):
                bounds_of_subphrases.append(('', (start_pos, len(src))))
            bounds_of_tokens = []
            for cur_subphrase in bounds_of_subphrases:
                if len(cur_subphrase[0]) == 0:
                    text = src[cur_subphrase[1][0]:cur_subphrase[1][1]]
                    bounds_of_tokens_in_text = self.__tokenize_text(text)
                    for cur_token in bounds_of_tokens_in_text:
                        bounds_of_tokens.append(
                            (
                                cur_subphrase[1][0] + cur_token[0],
                                cur_subphrase[1][0] + cur_token[1]
                            )
                        )
                else:
                    bounds_of_tokens.append(
                        (
                            cur_subphrase[1][0],
                            cur_subphrase[1][1]
                        )
                    )
        return bounds_of_tokens

    def __tokenize_text(self, src: str) -> List[Tuple[int, int]]:
        prep = src.strip()
        if len(prep) == 0:
            return []
        bounds_of_tokens = []
        end_pos = 0
        for cur_token in filter(lambda it2: len(it2) > 0,
                                map(lambda it1: it1.strip(), self.tokenizer.international_tokenize(src))):
            start_pos = src.find(cur_token, end_pos)
            if start_pos < 0:
                raise ValueError('Token `{0}` cannot be found in the text `{1}`!'.format(cur_token, src))
            end_pos = start_pos + len(cur_token)
            bounds_of_tokens.append((start_pos, end_pos))
        return bounds_of_tokens

    def __getstate__(self):
        return {'special_symbols': self.special_symbols}

    def __setstate__(self, state):
        self.special_symbols = state['special_symbols']
        if (self.special_symbols is not None) and (len(self.special_symbols) > 0):
            re_expr = '(' + '|'.join([re.escape(cur) for cur in self.special_symbols]) + ')'
            self.re_for_special_symbols = re.compile(re_expr)
        else:
            self.re_for_special_symbols = None
        self.tokenizer = NISTTokenizer()

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.special_symbols = self.special_symbols
        result.tokenizer = NISTTokenizer()

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.special_symbols = copy.deepcopy(self.special_symbols)
        result.tokenizer = NISTTokenizer()