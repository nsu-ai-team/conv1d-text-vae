import os
import pickle
import unittest

from annoy import AnnoyIndex
from gensim.models.keyedvectors import FastTextKeyedVectors
import numpy as np

from conv1d_text_vae.conv1d_text_vae import Conv1dTextVAE
from conv1d_text_vae.fasttext_loading import load_russian_fasttext_rusvectores
from conv1d_text_vae.sentence_reconstructor import SentenceReconstructor
from conv1d_text_vae.tokenizer import DefaultTokenizer


class TestSentenceReconstructor(unittest.TestCase):
    fasttext_model = None

    @classmethod
    def setUpClass(cls):
        cls.fasttext_model = load_russian_fasttext_rusvectores()

    @classmethod
    def tearDownClass(cls):
        del cls.fasttext_model

    def setUp(self):
        self.tmp_name = Conv1dTextVAE.get_temp_name()
        self.source_texts = [
            'Мама мыла раму',
            'Папа мыл синхрофазотрон',
            'Сын мыл машину',
            'Дочка мыла внучку'
        ]
        self.tokenizer = DefaultTokenizer()

    def tearDown(self):
        if os.path.isfile(self.tmp_name):
            os.remove(self.tmp_name)
        del self.source_texts
        del self.tokenizer

    def test_create(self):
        res = SentenceReconstructor(self.fasttext_model, self.tokenizer)
        self.assertIsInstance(res, SentenceReconstructor)
        self.assertTrue(hasattr(res, 'tokenizer'))
        self.assertTrue(hasattr(res, 'fasttext_vectors'))
        self.assertTrue(hasattr(res, 'special_symbols'))
        self.assertTrue(hasattr(res, 'n_variants'))
        self.assertFalse(hasattr(res, 'vocabulary_'))
        self.assertFalse(hasattr(res, 'annoy_index_'))
        self.assertFalse(hasattr(res, 'word_vector_size_'))

    def test_fit(self):
        reconstructor = SentenceReconstructor(self.fasttext_model, self.tokenizer)
        res = reconstructor.fit(self.source_texts)
        self.assertIsInstance(res, SentenceReconstructor)
        self.assertTrue(hasattr(res, 'tokenizer'))
        self.assertTrue(hasattr(res, 'fasttext_vectors'))
        self.assertTrue(hasattr(res, 'special_symbols'))
        self.assertTrue(hasattr(res, 'n_variants'))
        self.assertTrue(hasattr(res, 'vocabulary_'))
        self.assertTrue(hasattr(res, 'annoy_index_'))
        self.assertTrue(hasattr(res, 'word_vector_size_'))
        self.assertIsInstance(res.fasttext_vectors, FastTextKeyedVectors)
        self.assertIsInstance(res.annoy_index_, AnnoyIndex)

    def test_serialize_unfitted(self):
        res = SentenceReconstructor(self.fasttext_model, self.tokenizer)
        self.assertIsInstance(res, SentenceReconstructor)
        self.assertTrue(hasattr(res, 'tokenizer'))
        self.assertTrue(hasattr(res, 'fasttext_vectors'))
        self.assertTrue(hasattr(res, 'special_symbols'))
        self.assertTrue(hasattr(res, 'n_variants'))
        self.assertFalse(hasattr(res, 'vocabulary_'))
        self.assertFalse(hasattr(res, 'annoy_index_'))
        self.assertFalse(hasattr(res, 'word_vector_size_'))
        self.assertIsInstance(res.fasttext_vectors, FastTextKeyedVectors)
        with open(self.tmp_name, 'wb') as fp:
            pickle.dump(res, fp)
        with open(self.tmp_name, 'rb') as fp:
            res2 = pickle.load(fp)
        self.assertIsInstance(res2, SentenceReconstructor)
        self.assertIsNot(res, res2)
        self.assertTrue(hasattr(res2, 'tokenizer'))
        self.assertTrue(hasattr(res2, 'fasttext_vectors'))
        self.assertTrue(hasattr(res2, 'special_symbols'))
        self.assertTrue(hasattr(res2, 'n_variants'))
        self.assertFalse(hasattr(res2, 'vocabulary_'))
        self.assertFalse(hasattr(res2, 'annoy_index_'))
        self.assertFalse(hasattr(res2, 'word_vector_size_'))
        self.assertIsNone(res2.fasttext_vectors)
        self.assertEqual(res.special_symbols, res2.special_symbols)
        self.assertEqual(res.n_variants, res2.n_variants)

    def test_serialize_fitted(self):
        res = SentenceReconstructor(self.fasttext_model, self.tokenizer)
        self.assertIsInstance(res, SentenceReconstructor)
        self.assertTrue(hasattr(res, 'tokenizer'))
        self.assertTrue(hasattr(res, 'fasttext_vectors'))
        self.assertTrue(hasattr(res, 'special_symbols'))
        self.assertTrue(hasattr(res, 'n_variants'))
        res.fit(self.source_texts)
        self.assertTrue(hasattr(res, 'vocabulary_'))
        self.assertTrue(hasattr(res, 'annoy_index_'))
        self.assertTrue(hasattr(res, 'word_vector_size_'))
        with open(self.tmp_name, 'wb') as fp:
            pickle.dump(res, fp)
        with open(self.tmp_name, 'rb') as fp:
            res2 = pickle.load(fp)
        self.assertIsInstance(res2, SentenceReconstructor)
        self.assertIsNot(res, res2)
        self.assertTrue(hasattr(res2, 'tokenizer'))
        self.assertTrue(hasattr(res2, 'fasttext_vectors'))
        self.assertTrue(hasattr(res2, 'special_symbols'))
        self.assertTrue(hasattr(res2, 'n_variants'))
        self.assertTrue(hasattr(res2, 'vocabulary_'))
        self.assertTrue(hasattr(res2, 'annoy_index_'))
        self.assertTrue(hasattr(res2, 'word_vector_size_'))
        self.assertIsNone(res2.fasttext_vectors)
        self.assertEqual(res.special_symbols, res2.special_symbols)
        self.assertEqual(res.n_variants, res2.n_variants)
        self.assertEqual(res.word_vector_size_, res2.word_vector_size_)
        self.assertEqual(set(res.vocabulary_.keys()), set(res2.vocabulary_.keys()))
        for idx in res.vocabulary_:
            self.assertEqual(res.vocabulary_[idx], res2.vocabulary_[idx])
        self.assertIsInstance(res2.annoy_index_, AnnoyIndex)

    def test_transform(self):
        res = SentenceReconstructor(self.fasttext_model, self.tokenizer)
        res.fit(self.source_texts)
        vocabulary, word_vectors = Conv1dTextVAE.prepare_vocabulary_and_word_vectors(
            tuple([
                Conv1dTextVAE.tokenize(cur, self.tokenizer.tokenize_into_words(cur))
                for cur in self.source_texts
            ]),
            self.fasttext_model,
            special_symbols=None
        )
        data = np.vstack(
            (
                np.reshape(word_vectors[vocabulary['мама']], newshape=(1, res.word_vector_size_)),
                np.reshape(word_vectors[vocabulary['мыла']], newshape=(1, res.word_vector_size_)),
                np.reshape(word_vectors[vocabulary['раму']], newshape=(1, res.word_vector_size_)),
                np.reshape(word_vectors[vocabulary['']], newshape=(1, res.word_vector_size_)),
                np.reshape(word_vectors[vocabulary['папа']], newshape=(1, res.word_vector_size_)),
                np.reshape(word_vectors[vocabulary['мыл']], newshape=(1, res.word_vector_size_)),
                np.reshape(word_vectors[vocabulary['синхрофазотрон']], newshape=(1, res.word_vector_size_))
            )
        )
        data = np.reshape(data, newshape=(1, data.shape[0], data.shape[1]))
        reconstructed = res.transform(data)
        self.assertIsInstance(reconstructed, tuple)
        self.assertEqual(len(reconstructed), 1)
        self.assertIsInstance(reconstructed[0], tuple)
        self.assertEqual(len(reconstructed[0]), res.n_variants)
        ok = False
        for variant_idx in range(res.n_variants):
            self.assertIsInstance(reconstructed[0][variant_idx], tuple)
            self.assertGreater(len(reconstructed[0][variant_idx]), 0)
            if reconstructed[0][variant_idx] == ('мама', 'мыла', 'раму'):
                ok = True
            for time_idx in range(len(reconstructed[0][variant_idx])):
                self.assertIsInstance(reconstructed[0][variant_idx][time_idx], str)
                self.assertGreater(len(reconstructed[0][variant_idx][time_idx]), 0)
        self.assertTrue(ok)


if __name__ == '__main__':
    unittest.main(verbosity=2)
