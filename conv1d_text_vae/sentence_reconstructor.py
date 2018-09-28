import math
import os

from annoy import AnnoyIndex
from gensim.models.keyedvectors import FastTextKeyedVectors
import numpy as np
from sklearn.utils.validation import check_is_fitted

from conv1d_text_vae.tokenizer import BaseTokenizer, DefaultTokenizer
from conv1d_text_vae.conv1d_text_vae import Conv1dTextVAE


class SentenceReconstructor:
    def __init__(self, fasttext_vectors: FastTextKeyedVectors, tokenizer: BaseTokenizer=None,
                 special_symbols: tuple=None, n_variants: int=3):
        self.tokenizer = tokenizer
        self.fasttext_vectors = fasttext_vectors
        self.special_symbols = special_symbols
        self.n_variants = n_variants

    def fit(self, sentences):
        if not hasattr(self, 'fasttext_vectors'):
            raise ValueError('The `fasttext_vectors` attribute is not found!')
        if self.fasttext_vectors is None:
            raise ValueError('The `fasttext_vectors` attribute is not defined!')
        if self.tokenizer is None:
            self.tokenizer = DefaultTokenizer()
        if hasattr(self.tokenizer, 'special_symbols'):
            if (self.tokenizer.special_symbols is None) or (len(self.tokenizer.special_symbols) == 0):
                special_symbols = None
            else:
                special_symbols = tuple(sorted(list(self.tokenizer.special_symbols)))
        else:
            special_symbols = None
        tokenized_sentences = tuple([
            Conv1dTextVAE.tokenize(cur_sentence, self.tokenizer.tokenize_into_words(cur_sentence))
            for cur_sentence in sentences
        ])
        vocabulary, word_vectors = Conv1dTextVAE.prepare_vocabulary_and_word_vectors(
            tokenized_sentences, self.fasttext_vectors, special_symbols)
        reversed_vocabulary = list()
        self.word_vector_size_ = Conv1dTextVAE.calc_vector_size(self.fasttext_vectors, special_symbols)
        self.annoy_index_ = AnnoyIndex(self.word_vector_size_, metric='dot')
        for cur_word in vocabulary:
            word_idx = vocabulary[cur_word]
            self.annoy_index_.add_item(word_idx, word_vectors[word_idx])
            reversed_vocabulary.append((word_idx, cur_word))
        self.annoy_index_.build(int(round(math.sqrt(len(vocabulary)))))
        self.vocabulary_ = dict(reversed_vocabulary)
        return self

    def transform(self, matrices_of_sentences: np.ndarray):
        check_is_fitted(self, ['vocabulary_', 'annoy_index_', 'word_vector_size_'])
        if not isinstance(matrices_of_sentences, np.ndarray):
            raise ValueError('The `matrices_of_sentences` parameter is wrong! Expected {0}, got {1}.'.format(
                type(np.array([1, 2])), type(matrices_of_sentences)))
        if matrices_of_sentences.ndim != 3:
            raise ValueError('The `matrices_of_sentences` parameter is wrong! Expected a 3-D array, '
                             'got a {0}-D one.'.format(matrices_of_sentences.ndim))
        reconstructed_sentences = []
        for sample_idx in range(matrices_of_sentences.shape[0]):
            variants_of_new_sentence = []
            for time_idx in range(matrices_of_sentences.shape[1]):
                indices_of_words, similarities = self.annoy_index_.get_nns_by_vector(
                    matrices_of_sentences[sample_idx][time_idx], self.n_variants, include_distances=True)
                print('indices_of_words', indices_of_words)
                print('similarities', similarities)
                if self.vocabulary_[indices_of_words[0]] == '':
                    break
                variants_of_new_sentence.append(
                    tuple(filter(
                        lambda it1: it1[0] != '',
                        [(self.vocabulary_[indices_of_words[idx]], max(min(similarities[idx], 1.0), 0.0))
                         for idx in range(self.n_variants)]
                    ))
                )
            reconstructed_sentences.append(self.beam_search_decoder(variants_of_new_sentence, self.n_variants))
        return tuple(reconstructed_sentences)

    def __getstate__(self):
        state = {'tokenizer': self.tokenizer, 'special_symbols': self.special_symbols, 'n_variants': self.n_variants}
        if hasattr(self, 'vocabulary_') and hasattr(self, 'annoy_index_') and hasattr(self, 'word_vector_size_'):
            tmp_annoy_index_name = Conv1dTextVAE.get_temp_name()
            try:
                if os.path.isfile(tmp_annoy_index_name):
                    os.remove(tmp_annoy_index_name)
                self.annoy_index_.save(tmp_annoy_index_name)
                with open(tmp_annoy_index_name, 'rb') as fp:
                    annoy_index_data = fp.read()
                os.remove(tmp_annoy_index_name)
            finally:
                if os.path.isfile(tmp_annoy_index_name):
                    os.remove(tmp_annoy_index_name)
            state['annoy_index_'] = annoy_index_data
            state['vocabulary_'] = self.vocabulary_
            state['word_vector_size_'] = self.word_vector_size_
        return state

    def __setstate__(self, state):
        self.tokenizer = state['tokenizer']
        self.special_symbols = state['special_symbols']
        self.n_variants = state['n_variants']
        self.fasttext_vectors = None
        if hasattr(self, 'vocabulary_'):
            del self.vocabulary_
        if hasattr(self, 'annoy_index_'):
            del self.annoy_index_
        if ('annoy_index_' in state) and ('vocabulary_' in state) and ('word_vector_size_' in state):
            self.word_vector_size_ = state['word_vector_size_']
            tmp_annoy_index_name = Conv1dTextVAE.get_temp_name()
            try:
                with open(tmp_annoy_index_name, 'wb') as fp:
                    fp.write(state['annoy_index_'])
                self.annoy_index_ = AnnoyIndex(self.word_vector_size_)
                self.annoy_index_.load(tmp_annoy_index_name)
            finally:
                if os.path.isfile(tmp_annoy_index_name):
                    os.remove(tmp_annoy_index_name)
            self.vocabulary_ = state['vocabulary_']

    @staticmethod
    def beam_search_decoder(variants_of_sentence, n_best):
        EPS = 1e-9
        sequences = [[list(), 0.0]]
        for variants_of_word in variants_of_sentence:
            all_candidates = list()
            for idx1 in range(len(sequences)):
                seq, score = sequences[idx1]
                for idx2 in range(len(variants_of_word)):
                    candidate = [seq + [variants_of_word[idx2][0]], score - math.log(variants_of_word[idx2][1] + EPS)]
                    all_candidates.append(candidate)
            ordered = sorted(all_candidates, key=lambda it: it[1])
            sequences = ordered[:n_best]
        return tuple([tuple(it[0]) for it in sequences])