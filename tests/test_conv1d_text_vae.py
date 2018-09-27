import codecs
import copy
import csv
import os
import pickle
import re
import unittest

import numpy as np
from sklearn.exceptions import NotFittedError
from gensim.models.keyedvectors import FastTextKeyedVectors

from conv1d_text_vae.conv1d_text_vae import DefaultTokenizer, Conv1dTextVAE, SequenceForVAE
from conv1d_text_vae.fasttext_loading import load_russian_fasttext_rusvectores


class TestDefaultTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = DefaultTokenizer()
        self.name_of_tokenizer = os.path.join(os.path.dirname(__file__), 'testdata', 'tokenizer.pkl')

    def tearDown(self):
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if os.path.isfile(self.name_of_tokenizer):
            os.remove(self.name_of_tokenizer)

    def test_tokenize_into_words_positive01(self):
        s = ' Мама  мыла23\tраму.'
        true_bounds = [(1, 5), (7, 11), (11, 13), (14, 18), (18, 19)]
        predicted_bounds = self.tokenizer.tokenize_into_words(s)
        self.assertEqual(true_bounds, predicted_bounds)

    def test_tokenize_into_words_positive02(self):
        del self.tokenizer
        self.tokenizer = DefaultTokenizer(special_symbols={'\\n'})
        s = ' Мама  мыла23\\nраму.'
        true_bounds = [(1, 5), (7, 11), (11, 13), (13, 15), (15, 19), (19, 20)]
        predicted_bounds = self.tokenizer.tokenize_into_words(s)
        self.assertEqual(true_bounds, predicted_bounds)

    def test_tokenize_into_characters(self):
        s = ' Мама  мыла23 \n\tраму.'
        bounds_of_words = [(1, 5), (7, 11), (11, 13), (16, 20), (20, 21)]
        true_characters = [' ', 'М', 'а', 'м', 'а', ' ', 'м', 'ы', 'л', 'а', '2', '3', '\n', 'р', 'а', 'м', 'у', '.']
        predicted_characters = self.tokenizer.tokenize_into_characters(s, bounds_of_words)
        self.assertEqual(true_characters, predicted_characters)

    def test_serialize(self):
        with open(self.name_of_tokenizer, 'wb') as fp:
            pickle.dump(self.tokenizer, fp)
        with open(self.name_of_tokenizer, 'rb') as fp:
            res = pickle.load(fp)
        self.assertIsInstance(res, DefaultTokenizer)
        self.assertIsNot(res, self.tokenizer)
        s = ' Мама  мыла23\tраму.'
        self.assertEqual(self.tokenizer.tokenize_into_words(s), res.tokenize_into_words(s))


class TextSequenceForVAE(unittest.TestCase):
    fasttext_model = None

    @classmethod
    def setUpClass(cls):
        cls.fasttext_model = load_russian_fasttext_rusvectores()

    @classmethod
    def tearDownClass(cls):
        del cls.fasttext_model

    def test_positive01(self):
        src_data = [
            'как определить тип личности по форме носа: метод аристотеля',
            'какие преступления вдохновили достоевского',
            'майк тайсон - о пользе чтения'
        ]
        batch_size = 2
        input_text_size = 10
        output_text_size = 9
        tokenizer = DefaultTokenizer()
        input_texts = tuple(map(
            lambda it: Conv1dTextVAE.tokenize(it, tokenizer.tokenize_into_words(it)),
            src_data
        ))
        vocabulary, word_vectors = Conv1dTextVAE.prepare_vocabulary_and_word_vectors(
            input_texts,
            self.fasttext_model,
            None
        )
        generator = SequenceForVAE(tokenizer=tokenizer, input_texts=input_texts, target_texts=input_texts,
                                   batch_size=batch_size, input_text_size=input_text_size,
                                   output_text_size=output_text_size, input_vocabulary=vocabulary,
                                   output_vocabulary=vocabulary, input_word_vectors=word_vectors,
                                   output_word_vectors=word_vectors)
        true_length = 1
        predicted_length = len(generator)
        self.assertEqual(true_length, predicted_length)
        text_idx = 0
        for batch_idx in range(true_length):
            generated_data = generator[batch_idx]
            self.assertIsInstance(generated_data, tuple, msg='batch_idx={0}'.format(batch_idx))
            self.assertEqual(len(generated_data), 2, msg='batch_idx={0}'.format(batch_idx))
            self.assertIsInstance(generated_data[0], np.ndarray, msg='batch_idx={0}'.format(batch_idx))
            self.assertEqual(len(generated_data[0].shape), 3, msg='batch_idx={0}'.format(batch_idx))
            self.assertIsInstance(generated_data[1], np.ndarray, msg='batch_idx={0}'.format(batch_idx))
            self.assertEqual(len(generated_data[1].shape), 3, msg='batch_idx={0}'.format(batch_idx))
            self.assertEqual((batch_size, input_text_size, self.fasttext_model.vector_size + 2),
                             generated_data[0].shape, msg='batch_idx={0}'.format(batch_idx))
            self.assertEqual((batch_size, output_text_size, word_vectors.shape[0]),
                             generated_data[1].shape, msg='batch_idx={0}'.format(batch_idx))
            for sample_idx in range(batch_size):
                n_tokens = len(input_texts[text_idx])
                for token_idx in range(input_text_size):
                    vector_norm = np.linalg.norm(generated_data[0][sample_idx][token_idx])
                    if token_idx <= n_tokens:
                        self.assertAlmostEqual(vector_norm, 1.0, delta=1e-4,
                                               msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx))
                        if token_idx < n_tokens:
                            self.assertAlmostEqual(
                                generated_data[0][sample_idx][token_idx][self.fasttext_model.vector_size + 1],
                                0.0,
                                msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx)
                            )
                        else:
                            self.assertAlmostEqual(
                                generated_data[0][sample_idx][token_idx][self.fasttext_model.vector_size + 1],
                                1.0,
                                msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx)
                            )
                    else:
                        self.assertAlmostEqual(vector_norm, 0.0, delta=1e-4,
                                               msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx))
                for token_idx in range(output_text_size):
                    if token_idx <= n_tokens:
                        self.assertAlmostEqual(np.linalg.norm(generated_data[1][sample_idx][token_idx]), 1.0,
                                               msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx))
                        max_word_idx = np.argmax(generated_data[1][sample_idx][token_idx])
                        self.assertAlmostEqual(generated_data[1][sample_idx][token_idx][max_word_idx], 1.0,
                                               msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx))
                        if token_idx < n_tokens:
                            self.assertLess(max_word_idx, vocabulary[''],
                                            msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx))
                        else:
                            self.assertEqual(
                                max_word_idx,
                                vocabulary[''],
                                msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx)
                            )
                    else:
                        self.assertAlmostEqual(np.linalg.norm(generated_data[1][sample_idx][token_idx]), 0.0,
                                               msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx))
                if text_idx < (len(input_texts) - 1):
                    text_idx += 1

    def test_positive02(self):
        src_data = [
            'как определить тип личности\tпо форме носа:\nметод аристотеля',
            'какие преступления вдохновили достоевского',
            'майк тайсон - о пользе чтения'
        ]
        special_symbols = ('\t', '\n')
        batch_size = 2
        input_text_size = 10
        output_text_size = 9
        tokenizer = DefaultTokenizer(special_symbols=set(special_symbols))
        input_texts = tuple(map(
            lambda it: Conv1dTextVAE.tokenize(it, tokenizer.tokenize_into_words(it)),
            src_data
        ))
        vocabulary, word_vectors = Conv1dTextVAE.prepare_vocabulary_and_word_vectors(
            input_texts,
            self.fasttext_model,
            special_symbols
        )
        generator = SequenceForVAE(tokenizer=tokenizer, input_texts=input_texts, target_texts=input_texts,
                                   batch_size=batch_size, input_text_size=input_text_size,
                                   output_text_size=output_text_size, input_vocabulary=vocabulary,
                                   output_vocabulary=vocabulary, input_word_vectors=word_vectors,
                                   output_word_vectors=word_vectors)
        true_length = 1
        predicted_length = len(generator)
        self.assertEqual(true_length, predicted_length)
        text_idx = 0
        for batch_idx in range(true_length):
            generated_data = generator[batch_idx]
            self.assertIsInstance(generated_data, tuple, msg='batch_idx={0}'.format(batch_idx))
            self.assertEqual(len(generated_data), 2, msg='batch_idx={0}'.format(batch_idx))
            self.assertIsInstance(generated_data[0], np.ndarray, msg='batch_idx={0}'.format(batch_idx))
            self.assertEqual(len(generated_data[0].shape), 3, msg='batch_idx={0}'.format(batch_idx))
            self.assertIsInstance(generated_data[1], np.ndarray, msg='batch_idx={0}'.format(batch_idx))
            self.assertEqual(len(generated_data[1].shape), 3, msg='batch_idx={0}'.format(batch_idx))
            self.assertEqual((batch_size, input_text_size, self.fasttext_model.vector_size + 4),
                             generated_data[0].shape, msg='batch_idx={0}'.format(batch_idx))
            self.assertEqual((batch_size, output_text_size, word_vectors.shape[0]),
                             generated_data[1].shape, msg='batch_idx={0}'.format(batch_idx))
            for sample_idx in range(batch_size):
                tokens = input_texts[text_idx]
                n_tokens = len(tokens)
                for token_idx in range(input_text_size):
                    vector_norm = np.linalg.norm(generated_data[0][sample_idx][token_idx])
                    if token_idx <= n_tokens:
                        self.assertAlmostEqual(vector_norm, 1.0, delta=1e-4,
                                               msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx))
                        if token_idx < n_tokens:
                            self.assertAlmostEqual(
                                generated_data[0][sample_idx][token_idx][self.fasttext_model.vector_size + 3],
                                0.0,
                                msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx)
                            )
                            if tokens[token_idx] in special_symbols:
                                self.assertAlmostEqual(
                                    generated_data[0][sample_idx][token_idx][self.fasttext_model.vector_size + 1 +
                                                                             special_symbols.index(tokens[token_idx])],
                                    1.0,
                                    msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx)
                                )
                            else:
                                self.assertAlmostEqual(
                                    generated_data[0][sample_idx][token_idx][self.fasttext_model.vector_size + 1],
                                    0.0,
                                    msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx)
                                )
                                self.assertAlmostEqual(
                                    generated_data[0][sample_idx][token_idx][self.fasttext_model.vector_size + 2],
                                    0.0,
                                    msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx)
                                )
                        else:
                            self.assertAlmostEqual(
                                generated_data[0][sample_idx][token_idx][self.fasttext_model.vector_size + 3],
                                1.0,
                                msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx)
                            )
                    else:
                        self.assertAlmostEqual(vector_norm, 0.0, delta=1e-4,
                                               msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx))
                for token_idx in range(output_text_size):
                    if token_idx <= n_tokens:
                        self.assertAlmostEqual(np.linalg.norm(generated_data[1][sample_idx][token_idx]), 1.0,
                                               msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx))
                        max_word_idx = np.argmax(generated_data[1][sample_idx][token_idx])
                        self.assertAlmostEqual(generated_data[1][sample_idx][token_idx][max_word_idx], 1.0,
                                               msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx))
                        if token_idx < n_tokens:
                            if tokens[token_idx] in special_symbols:
                                self.assertEqual(
                                    max_word_idx,
                                    vocabulary[tokens[token_idx]],
                                    msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx)
                                )
                            else:
                                self.assertLess(
                                    max_word_idx,
                                    vocabulary[''] - len(special_symbols),
                                    msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx)
                                )
                        else:
                            self.assertEqual(
                                max_word_idx,
                                vocabulary[''],
                                msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx)
                            )
                    else:
                        self.assertAlmostEqual(np.linalg.norm(generated_data[1][sample_idx][token_idx]), 0.0,
                                               msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx))
                if text_idx < (len(input_texts) - 1):
                    text_idx += 1


class FakeVAE():
    def predict(self, x):
        return x

    def predict_on_batch(self, x):
        return x


class TestConv1dTextVAE(unittest.TestCase):
    ru_fasttext_model = None

    @classmethod
    def setUpClass(cls):
        cls.ru_fasttext_model = load_russian_fasttext_rusvectores()
        text_corpus_name = os.path.join(os.path.dirname(__file__), 'testdata', 'testdata.csv')
        cls.input_texts = []
        cls.target_texts = []
        with codecs.open(text_corpus_name, mode='r', encoding='utf-8', errors='ignore') as fp:
            data_reader = csv.reader(fp, delimiter=',', quotechar='"')
            header = None
            true_header = ['annotation', 'full_text']
            for row in data_reader:
                if len(row) > 0:
                    if header is None:
                        header = copy.copy(row)
                        assert header == true_header
                    else:
                        assert len(row) == len(header)
                        cls.input_texts.append(row[0].strip().lower())
                        cls.target_texts.append(row[1].strip().lower())

    @classmethod
    def tearDownClass(cls):
        del cls.ru_fasttext_model

    def setUp(self):
        self.text_vae = Conv1dTextVAE(
            input_embeddings=self.ru_fasttext_model, output_embeddings=self.ru_fasttext_model,
            max_epochs=10, batch_size=2
        )
        self.model_name = os.path.join(os.path.dirname(__file__), 'testdata', 'model.pkl')

    def tearDown(self):
        if hasattr(self, 'text_vae'):
            del self.text_vae
        if os.path.isfile(self.model_name):
            os.remove(self.model_name)

    def test_create(self):
        self.assertIsInstance(self.text_vae, Conv1dTextVAE)
        self.assertTrue(hasattr(self.text_vae, 'n_filters'))
        self.assertTrue(hasattr(self.text_vae, 'kernel_size'))
        self.assertTrue(hasattr(self.text_vae, 'input_embeddings'))
        self.assertTrue(hasattr(self.text_vae, 'output_embeddings'))
        self.assertTrue(hasattr(self.text_vae, 'batch_size'))
        self.assertTrue(hasattr(self.text_vae, 'max_epochs'))
        self.assertTrue(hasattr(self.text_vae, 'latent_dim'))
        self.assertTrue(hasattr(self.text_vae, 'max_dist_between_output_synonyms'))
        self.assertTrue(hasattr(self.text_vae, 'use_batch_norm'))
        self.assertTrue(hasattr(self.text_vae, 'warm_start'))
        self.assertTrue(hasattr(self.text_vae, 'verbose'))
        self.assertTrue(hasattr(self.text_vae, 'input_text_size'))
        self.assertTrue(hasattr(self.text_vae, 'output_text_size'))
        self.assertTrue(hasattr(self.text_vae, 'validation_fraction'))
        self.assertTrue(hasattr(self.text_vae, 'tokenizer'))
        self.assertFalse(hasattr(self.text_vae, 'input_text_size_'))
        self.assertFalse(hasattr(self.text_vae, 'output_text_size_'))
        self.assertFalse(hasattr(self.text_vae, 'input_vector_size_'))
        self.assertFalse(hasattr(self.text_vae, 'output_vector_size_'))
        self.assertFalse(hasattr(self.text_vae, 'vae_encoder_'))
        self.assertFalse(hasattr(self.text_vae, 'vae_decoder_'))

    def test_tokenize(self):
        tokenizer = DefaultTokenizer(special_symbols={'\n'})
        s = ' Мама  мыла23\nраму.'
        true_tokens = ('мама', 'мыла', '23', '\n', 'раму', '.')
        predicted_tokens = Conv1dTextVAE.tokenize(s, tokenizer.tokenize_into_words(s))
        self.assertIsInstance(predicted_tokens, tuple)
        self.assertEqual(true_tokens, predicted_tokens)

    def test_prepare_vocabulary_and_word_vectors(self):
        eps = 1e-5
        input_texts = (
            ('мама', 'мыла', 'раму'),
            ('папа', 'мыл', 'синхрофазотрон', '!'),
            ('а', 'дочка', 'мыла', 'свой', 'ноутбук', 'от', 'коньяка')
        )
        true_words_in_vocabulary = {'мама', 'мыла', 'раму', 'папа', 'мыл', 'синхрофазотрон', '!', 'а', 'дочка', 'мыла',
                                    'свой', 'ноутбук', 'от', 'коньяка', ''}
        true_vector_size = self.ru_fasttext_model.vector_size + 3
        special_symbols = ('!',)
        calc_vocabulary, calc_word_vectors = Conv1dTextVAE.prepare_vocabulary_and_word_vectors(
            input_texts, self.ru_fasttext_model, special_symbols)
        self.assertIsInstance(calc_vocabulary, dict)
        self.assertEqual(true_words_in_vocabulary, set(calc_vocabulary.keys()))
        self.assertEqual(calc_vocabulary[''], len(true_words_in_vocabulary) - 1)
        for cur_word in true_words_in_vocabulary - {''}:
            self.assertIn(cur_word, calc_vocabulary)
            self.assertGreaterEqual(calc_vocabulary[cur_word], 0)
            self.assertLess(calc_vocabulary[cur_word], len(true_words_in_vocabulary) - 1)
        self.assertIsInstance(calc_word_vectors, np.ndarray)
        self.assertEqual((len(true_words_in_vocabulary), true_vector_size), calc_word_vectors.shape)
        special_word_indices = {len(true_words_in_vocabulary) - 1, calc_vocabulary['!']}
        for word_idx in range(calc_word_vectors.shape[0]):
            vector_norm = np.linalg.norm(calc_word_vectors[word_idx])
            self.assertAlmostEqual(vector_norm, 1.0, places=4)
            if word_idx not in special_word_indices:
                self.assertAlmostEqual(calc_word_vectors[word_idx, true_vector_size - 1], 0.0)
                self.assertAlmostEqual(calc_word_vectors[word_idx, true_vector_size - 2], 0.0)
                self.assertTrue((abs(calc_word_vectors[word_idx, true_vector_size - 3]) < eps) or \
                                (abs(calc_word_vectors[word_idx, true_vector_size - 3] - 1.0) < eps))
        self.assertAlmostEqual(calc_word_vectors[len(true_words_in_vocabulary) - 1, true_vector_size - 1], 1.0)
        self.assertAlmostEqual(calc_word_vectors[calc_vocabulary['!'], true_vector_size - 2], 1.0)

    def test_get_vocabulary_and_word_vectors_from_fasttext(self):
        input_texts = (
            ('мама', 'мыла', 'раму'),
            ('папа', 'мыл', 'синхрофазотрон', '!'),
            ('а', 'дочка', 'мыла', 'свой', 'ноутбук', 'от', 'коньяка')
        )
        true_words_in_vocabulary = {'мама', 'мыла', 'раму', 'папа', 'мыл', 'синхрофазотрон', '!', 'а', 'дочка', 'мыла',
                                    'свой', 'ноутбук', 'от', 'коньяка', ''}
        true_vector_size = self.ru_fasttext_model.vector_size
        special_symbols = ('мама',)
        calc_vocabulary, calc_word_vectors = Conv1dTextVAE.get_vocabulary_and_word_vectors_from_fasttext(
            input_texts, self.ru_fasttext_model, special_symbols)
        self.assertIsInstance(calc_vocabulary, dict)
        self.assertLess(set(calc_vocabulary.keys()), true_words_in_vocabulary)
        self.assertFalse(special_symbols[0] in calc_vocabulary)
        self.assertIsInstance(calc_word_vectors, np.ndarray)
        self.assertEqual((len(calc_vocabulary), true_vector_size), calc_word_vectors.shape)
        used_indices = set()
        for cur_word in calc_vocabulary:
            idx = calc_vocabulary[cur_word]
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, calc_word_vectors.shape[0])
            self.assertFalse(idx in used_indices)
            used_indices.add(idx)
            vector_norm = np.linalg.norm(calc_word_vectors[idx])
            self.assertAlmostEqual(vector_norm, 1.0, places=4)

    def test_check_texts_param_negative01(self):
        true_err_msg = re.escape('The parameter `X` is wrong! Expected `{0}`, `{1}` or 1-D `{2}`, got `{3}`.'.format(
            type([1, 2]), type((1, 2)),type(np.array([1, 2])), type({1, 2})))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_texts_param(set(self.input_texts), 'X')

    def test_check_texts_param_negative02(self):
        texts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        true_err_msg = re.escape('The parameter `y` is wrong! Expected 1-D array, got {0}-D array.'.format(
            len(texts.shape)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_texts_param(texts, 'y')

    def test_check_texts_param_negative03(self):
        texts = copy.deepcopy(self.input_texts)
        texts[3] = 4.5
        true_err_msg = re.escape('Item 3 of the parameter `X` is wrong! This item is not string!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_texts_param(texts, 'X')

    def test_check_params_negative01(self):
        params = self.text_vae.__dict__
        del params['warm_start']
        true_err_msg = re.escape('The parameter `warm_start` is not defined!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative02(self):
        params = self.text_vae.__dict__
        params['warm_start'] = 0.5
        true_err_msg = re.escape('The parameter `warm_start` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(True), type(0.5)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative03(self):
        params = self.text_vae.__dict__
        del params['verbose']
        true_err_msg = re.escape('The parameter `verbose` is not defined!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative04(self):
        params = self.text_vae.__dict__
        params['verbose'] = 0.5
        true_err_msg = re.escape('The parameter `verbose` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(True), type(0.5)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative05(self):
        params = self.text_vae.__dict__
        del params['batch_size']
        true_err_msg = re.escape('The parameter `batch_size` is not defined!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative06(self):
        params = self.text_vae.__dict__
        params['batch_size'] = 4.5
        true_err_msg = re.escape('The parameter `batch_size` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(10), type(4.5)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative07(self):
        params = self.text_vae.__dict__
        params['batch_size'] = -3
        true_err_msg = re.escape('The parameter `batch_size` is wrong! Expected a positive value, '
                                 'but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative08(self):
        params = self.text_vae.__dict__
        del params['max_epochs']
        true_err_msg = re.escape('The parameter `max_epochs` is not defined!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative09(self):
        params = self.text_vae.__dict__
        params['max_epochs'] = 4.5
        true_err_msg = re.escape('The parameter `max_epochs` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(10), type(4.5)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative10(self):
        params = self.text_vae.__dict__
        params['max_epochs'] = -3
        true_err_msg = re.escape('The parameter `max_epochs` is wrong! Expected a positive value, '
                                 'but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative11(self):
        params = self.text_vae.__dict__
        del params['latent_dim']
        true_err_msg = re.escape('The parameter `latent_dim` is not defined!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative12(self):
        params = self.text_vae.__dict__
        params['latent_dim'] = 4.5
        true_err_msg = re.escape('The parameter `latent_dim` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(10), type(4.5)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative13(self):
        params = self.text_vae.__dict__
        params['latent_dim'] = -3
        true_err_msg = re.escape('The parameter `latent_dim` is wrong! Expected a positive value, '
                                 'but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative14(self):
        params = self.text_vae.__dict__
        del params['input_text_size']
        true_err_msg = re.escape('The parameter `input_text_size` is not defined!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative15(self):
        params = self.text_vae.__dict__
        params['input_text_size'] = 4.5
        true_err_msg = re.escape('The parameter `input_text_size` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(10), type(4.5)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative16(self):
        params = self.text_vae.__dict__
        params['input_text_size'] = -3
        true_err_msg = re.escape('The parameter `input_text_size` is wrong! Expected a positive value, '
                                 'but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative17(self):
        params = self.text_vae.__dict__
        del params['output_text_size']
        true_err_msg = re.escape('The parameter `output_text_size` is not defined!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative18(self):
        params = self.text_vae.__dict__
        params['output_text_size'] = 4.5
        true_err_msg = re.escape('The parameter `output_text_size` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(10), type(4.5)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative19(self):
        params = self.text_vae.__dict__
        params['output_text_size'] = -3
        true_err_msg = re.escape('The parameter `output_text_size` is wrong! Expected a positive value, '
                                 'but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative20(self):
        params = self.text_vae.__dict__
        del params['n_filters']
        true_err_msg = re.escape('The parameter `n_filters` is not defined!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative21(self):
        params = self.text_vae.__dict__
        params['n_filters'] = 4.5
        true_err_msg = re.escape('The parameter `n_filters` is wrong! Expected `{0}` or `{1}`, got `{2}`.'.format(
            type(10), type((1, 2)), type(4.5)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative22(self):
        params = self.text_vae.__dict__
        params['n_filters'] = -3
        true_err_msg = re.escape('The parameter `n_filters` is wrong! Expected a positive value, '
                                 'but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative23(self):
        params = self.text_vae.__dict__
        params['n_filters'] = (10, -3)
        true_err_msg = re.escape('Item 1 of the parameter `n_filters` is wrong! Expected a positive value, '
                                 'but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative24(self):
        params = self.text_vae.__dict__
        del params['kernel_size']
        true_err_msg = re.escape('The parameter `kernel_size` is not defined!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative25(self):
        params = self.text_vae.__dict__
        params['kernel_size'] = 4.5
        true_err_msg = re.escape('The parameter `kernel_size` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(10), type(4.5)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative26(self):
        params = self.text_vae.__dict__
        params['kernel_size'] = -3
        true_err_msg = re.escape('The parameter `kernel_size` is wrong! Expected a positive value, '
                                 'but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative27(self):
        params = self.text_vae.__dict__
        del params['validation_fraction']
        true_err_msg = re.escape('The parameter `validation_fraction` is not defined!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative28(self):
        params = self.text_vae.__dict__
        params['validation_fraction'] = '1.5'
        true_err_msg = re.escape('The parameter `validation_fraction` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(10.5), type('1.5')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative29(self):
        params = self.text_vae.__dict__
        params['validation_fraction'] = -0.1
        true_err_msg = re.escape('The parameter `validation_fraction` is wrong! Expected a positive value between 0.0 '
                                 'and 1.0, but -0.1 does not correspond to this condition.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative30(self):
        params = self.text_vae.__dict__
        del params['use_batch_norm']
        true_err_msg = re.escape('The parameter `use_batch_norm` is not defined!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative31(self):
        params = self.text_vae.__dict__
        del params['output_onehot_size']
        true_err_msg = re.escape('The parameter `output_onehot_size` is not defined!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative32(self):
        params = self.text_vae.__dict__
        params['output_onehot_size'] = 4.5
        true_err_msg = re.escape('The parameter `output_onehot_size` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(10), type(4.5)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative33(self):
        params = self.text_vae.__dict__
        params['output_onehot_size'] = -3
        true_err_msg = re.escape('The parameter `output_onehot_size` is wrong! Expected a positive value, '
                                 'but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative34(self):
        params = self.text_vae.__dict__
        del params['max_dist_between_output_synonyms']
        true_err_msg = re.escape('The parameter `max_dist_between_output_synonyms` is not defined!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative35(self):
        params = self.text_vae.__dict__
        params['max_dist_between_output_synonyms'] = '1.5'
        true_err_msg = re.escape('The parameter `max_dist_between_output_synonyms` is wrong! Expected `{0}`, '
                                 'got `{1}`.'.format(type(10.5), type('1.5')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_check_params_negative36(self):
        params = self.text_vae.__dict__
        params['max_dist_between_output_synonyms'] = -0.1
        true_err_msg = 'The parameter \`max_dist_between_output_synonyms\` is wrong\! Expected a non-negative ' \
                       'value\, but \-\d+\.\d+ is negative\.'
        with self.assertRaisesRegex(ValueError, true_err_msg):
            Conv1dTextVAE.check_params(**params)

    def test_float_to_string_positive01(self):
        value = 3.567
        true_res = '3.567'
        calc_res = Conv1dTextVAE.float_to_string(value)
        self.assertEqual(true_res, calc_res)

    def test_float_to_string_positive02(self):
        value = 0.50670
        true_res = '0.5067'
        calc_res = Conv1dTextVAE.float_to_string(value)
        self.assertEqual(true_res, calc_res)

    def test_texts_to_data_positive01(self):
        batch_size = 2
        input_text_size = 10
        tokenizer = DefaultTokenizer()
        true_length = 2
        n_texts = 3
        predicted_data = list(Conv1dTextVAE.texts_to_data(
            self.input_texts[0:n_texts], batch_size=2, max_text_size=10, tokenizer=tokenizer,
            fasttext_model=self.ru_fasttext_model
        ))
        predicted_length = len(predicted_data)
        self.assertEqual(true_length, predicted_length)
        text_idx = 0
        for batch_idx in range(true_length):
            batch_data = predicted_data[batch_idx]
            self.assertIsInstance(batch_data, np.ndarray, msg='batch_idx={0}'.format(batch_idx))
            self.assertEqual(len(batch_data.shape), 3, msg='batch_idx={0}'.format(batch_idx))
            self.assertEqual((batch_size, input_text_size, self.ru_fasttext_model.vector_size + 2),
                             batch_data.shape, msg='batch_idx={0}'.format(batch_idx))
            for sample_idx in range(batch_size):
                n_tokens = len(Conv1dTextVAE.tokenize(
                    self.input_texts[text_idx], tokenizer.tokenize_into_words(self.input_texts[text_idx])
                ))
                for token_idx in range(input_text_size):
                    vector_norm = np.linalg.norm(batch_data[sample_idx][token_idx])
                    if token_idx <= n_tokens:
                        self.assertAlmostEqual(vector_norm, 1.0, delta=1e-4,
                                               msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx))
                        if token_idx < n_tokens:
                            self.assertAlmostEqual(
                                batch_data[sample_idx][token_idx][self.ru_fasttext_model.vector_size + 1],
                                0.0,
                                msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx)
                            )
                        else:
                            self.assertAlmostEqual(
                                batch_data[sample_idx][token_idx][self.ru_fasttext_model.vector_size + 1],
                                1.0,
                                msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx)
                            )
                    else:
                        self.assertAlmostEqual(vector_norm, 0.0, delta=1e-4,
                                               msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx))
                if text_idx < (n_texts - 1):
                    text_idx += 1

    def test_texts_to_data_positive02(self):
        special_symbols = ('\t', '\n')
        batch_size = 2
        input_text_size = 10
        tokenizer = DefaultTokenizer()
        true_length = 2
        n_texts = 3
        input_texts = copy.deepcopy(self.input_texts[0:n_texts])
        input_texts[0].replace(' ', '\n', 1)
        input_texts[1].replace(' ', '\t', 1)
        predicted_data = list(Conv1dTextVAE.texts_to_data(
            input_texts, batch_size=2, max_text_size=10, tokenizer=tokenizer,
            fasttext_model=self.ru_fasttext_model, special_symbols=special_symbols
        ))
        predicted_length = len(predicted_data)
        self.assertEqual(true_length, predicted_length)
        text_idx = 0
        for batch_idx in range(true_length):
            batch_data = predicted_data[batch_idx]
            self.assertIsInstance(batch_data, np.ndarray, msg='batch_idx={0}'.format(batch_idx))
            self.assertEqual(len(batch_data.shape), 3, msg='batch_idx={0}'.format(batch_idx))
            self.assertEqual((batch_size, input_text_size, self.ru_fasttext_model.vector_size + 4),
                             batch_data.shape, msg='batch_idx={0}'.format(batch_idx))
            for sample_idx in range(batch_size):
                tokens = Conv1dTextVAE.tokenize(
                    input_texts[text_idx], tokenizer.tokenize_into_words(input_texts[text_idx])
                )
                n_tokens = len(tokens)
                for token_idx in range(input_text_size):
                    vector_norm = np.linalg.norm(batch_data[sample_idx][token_idx])
                    if token_idx <= n_tokens:
                        self.assertAlmostEqual(vector_norm, 1.0, delta=1e-4,
                                               msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx))
                        if token_idx < n_tokens:
                            self.assertAlmostEqual(
                                batch_data[sample_idx][token_idx][self.ru_fasttext_model.vector_size + 3],
                                0.0,
                                msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx)
                            )
                            if tokens[token_idx] in special_symbols:
                                self.assertAlmostEqual(
                                    batch_data[sample_idx][token_idx][self.ru_fasttext_model.vector_size +
                                                                      special_symbols.index(tokens[token_idx])],
                                    1.0,
                                    msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx)
                                )
                            else:
                                self.assertAlmostEqual(
                                    batch_data[sample_idx][token_idx][self.ru_fasttext_model.vector_size],
                                    0.0,
                                    msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx)
                                )
                                self.assertAlmostEqual(
                                    batch_data[sample_idx][token_idx][self.ru_fasttext_model.vector_size + 1],
                                    0.0,
                                    msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx)
                                )
                        else:
                            self.assertAlmostEqual(
                                batch_data[sample_idx][token_idx][self.ru_fasttext_model.vector_size + 3],
                                1.0,
                                msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx)
                            )
                    else:
                        self.assertAlmostEqual(vector_norm, 0.0, delta=1e-4,
                                               msg='batch_idx={0}, sample_idx={1}'.format(batch_idx, sample_idx))
                if text_idx < (n_texts - 1):
                    text_idx += 1

    def test_find_best_words_positive01(self):
        word_vector = np.zeros((self.ru_fasttext_model.vector_size + 2,), dtype=np.float32)
        word_vector[0:self.ru_fasttext_model.vector_size] = self.ru_fasttext_model['нейросеть']
        best_words = Conv1dTextVAE.find_best_words(word_vector, self.ru_fasttext_model, 3)
        self.assertIsInstance(best_words, list)
        self.assertEqual(len(best_words), 3)
        for variant_idx in range(len(best_words)):
            self.assertIsInstance(best_words[variant_idx], tuple)
            self.assertEqual(len(best_words[variant_idx]), 2)
            self.assertIsInstance(best_words[variant_idx][0], str)
            self.assertIsInstance(best_words[variant_idx][1], float)
            self.assertGreaterEqual(best_words[variant_idx][1], 0.0)
        for variant_idx in range(1, len(best_words)):
            self.assertLessEqual(best_words[variant_idx - 1][1], best_words[variant_idx][1])

    def test_find_best_words_positive02(self):
        word_vector = np.zeros((self.ru_fasttext_model.vector_size + 2,), dtype=np.float32)
        word_vector[self.ru_fasttext_model.vector_size + 1] = 1.0
        best_words = Conv1dTextVAE.find_best_words(word_vector, self.ru_fasttext_model, 3)
        self.assertIsNone(best_words)

    def test_find_best_words_positive03(self):
        word_vector = np.zeros((self.ru_fasttext_model.vector_size + 2,), dtype=np.float32)
        word_vector[self.ru_fasttext_model.vector_size] = 1.0
        best_words = Conv1dTextVAE.find_best_words(word_vector, self.ru_fasttext_model, 3)
        self.assertIsInstance(best_words, list)
        self.assertEqual(len(best_words), 0)

    def test_find_best_words_positive04(self):
        special_symbols = ('\t', '\n')
        word_vector = np.zeros((self.ru_fasttext_model.vector_size + 4,), dtype=np.float32)
        word_vector[self.ru_fasttext_model.vector_size + 1] = 1.0
        best_words = Conv1dTextVAE.find_best_words(word_vector, self.ru_fasttext_model, 3, special_symbols)
        self.assertIsInstance(best_words, list)
        self.assertEqual(len(best_words), 1)
        self.assertIsInstance(best_words[0], tuple)
        self.assertEqual(len(best_words[0]), 2)
        self.assertEqual(best_words[0][0], '\n')
        self.assertAlmostEqual(best_words[0][1], 0.0)

    def test_find_best_texts(self):
        word_vector_1 = np.zeros((self.ru_fasttext_model.vector_size + 2,), dtype=np.float32)
        word_vector_1[0:self.ru_fasttext_model.vector_size] = self.ru_fasttext_model['мама']
        word_vector_2 = np.zeros((self.ru_fasttext_model.vector_size + 2,), dtype=np.float32)
        word_vector_2[0:self.ru_fasttext_model.vector_size] = self.ru_fasttext_model['мыть']
        word_vector_3 = np.zeros((self.ru_fasttext_model.vector_size + 2,), dtype=np.float32)
        word_vector_3[0:self.ru_fasttext_model.vector_size] = self.ru_fasttext_model['рама']
        ntop = 5
        true_top_variant = 'мама мыть рама'
        best_variants_of_word1 = tuple(Conv1dTextVAE.find_best_words(word_vector_1, self.ru_fasttext_model, ntop))
        best_variants_of_word2 = tuple(Conv1dTextVAE.find_best_words(word_vector_2, self.ru_fasttext_model, ntop))
        best_variants_of_word3 = tuple(Conv1dTextVAE.find_best_words(word_vector_3, self.ru_fasttext_model, ntop))
        best_variants = Conv1dTextVAE.find_best_texts(
            [
                best_variants_of_word1,
                best_variants_of_word2,
                best_variants_of_word3
            ],
            ntop
        )
        self.assertIsInstance(best_variants, list)
        self.assertEqual(len(best_variants), ntop)
        for variant_idx in range(len(best_variants)):
            self.assertIsInstance(best_variants[variant_idx], str)
        self.assertEqual(best_variants[0], true_top_variant)

    def test_fit_positive01(self):
        self.text_vae.verbose = 2
        self.text_vae.use_attention = False
        res = self.text_vae.fit(self.input_texts, self.target_texts)
        self.assertIsInstance(res, Conv1dTextVAE)
        self.assertTrue(hasattr(res, 'n_filters'))
        self.assertTrue(hasattr(res, 'kernel_size'))
        self.assertTrue(hasattr(res, 'input_embeddings'))
        self.assertTrue(hasattr(res, 'output_embeddings'))
        self.assertTrue(hasattr(res, 'batch_size'))
        self.assertTrue(hasattr(res, 'max_epochs'))
        self.assertTrue(hasattr(res, 'latent_dim'))
        self.assertTrue(hasattr(res, 'max_dist_between_output_synonyms'))
        self.assertTrue(hasattr(res, 'use_batch_norm'))
        self.assertTrue(hasattr(res, 'warm_start'))
        self.assertTrue(hasattr(res, 'verbose'))
        self.assertTrue(hasattr(res, 'input_text_size'))
        self.assertTrue(hasattr(res, 'output_text_size'))
        self.assertTrue(hasattr(res, 'validation_fraction'))
        self.assertTrue(hasattr(res, 'tokenizer'))
        self.assertTrue(hasattr(res, 'input_text_size_'))
        self.assertTrue(hasattr(res, 'output_text_size_'))
        self.assertTrue(hasattr(res, 'input_vector_size_'))
        self.assertTrue(hasattr(res, 'output_vector_size_'))
        self.assertTrue(hasattr(res, 'vae_encoder_'))
        self.assertTrue(hasattr(res, 'vae_decoder_'))

    def test_fit_positive02(self):
        self.text_vae.verbose = 2
        self.text_vae.output_onehot_size = 3
        res = self.text_vae.fit(self.input_texts)
        self.assertIsInstance(res, Conv1dTextVAE)
        self.assertIsInstance(res, Conv1dTextVAE)
        self.assertTrue(hasattr(res, 'n_filters'))
        self.assertTrue(hasattr(res, 'kernel_size'))
        self.assertTrue(hasattr(res, 'input_embeddings'))
        self.assertTrue(hasattr(res, 'output_embeddings'))
        self.assertTrue(hasattr(res, 'batch_size'))
        self.assertTrue(hasattr(res, 'max_epochs'))
        self.assertTrue(hasattr(res, 'latent_dim'))
        self.assertTrue(hasattr(res, 'max_dist_between_output_synonyms'))
        self.assertTrue(hasattr(res, 'use_batch_norm'))
        self.assertTrue(hasattr(res, 'warm_start'))
        self.assertTrue(hasattr(res, 'verbose'))
        self.assertTrue(hasattr(res, 'input_text_size'))
        self.assertTrue(hasattr(res, 'output_text_size'))
        self.assertTrue(hasattr(res, 'validation_fraction'))
        self.assertTrue(hasattr(res, 'tokenizer'))
        self.assertTrue(hasattr(res, 'input_text_size_'))
        self.assertTrue(hasattr(res, 'output_text_size_'))
        self.assertTrue(hasattr(res, 'input_vector_size_'))
        self.assertTrue(hasattr(res, 'output_vector_size_'))
        self.assertTrue(hasattr(res, 'vae_encoder_'))
        self.assertTrue(hasattr(res, 'vae_decoder_'))

    def test_fit_positive03(self):
        self.text_vae.verbose = 2
        print('')
        print('PRE-FITTING')
        res = self.text_vae.fit(self.input_texts)
        res.warm_start = True
        print('')
        print('FINE-TUNING')
        res.fit(self.input_texts, self.target_texts)
        self.assertIsInstance(res, Conv1dTextVAE)
        self.assertTrue(hasattr(res, 'n_filters'))
        self.assertTrue(hasattr(res, 'kernel_size'))
        self.assertTrue(hasattr(res, 'input_embeddings'))
        self.assertTrue(hasattr(res, 'output_embeddings'))
        self.assertTrue(hasattr(res, 'batch_size'))
        self.assertTrue(hasattr(res, 'max_epochs'))
        self.assertTrue(hasattr(res, 'latent_dim'))
        self.assertTrue(hasattr(res, 'max_dist_between_output_synonyms'))
        self.assertTrue(hasattr(res, 'use_batch_norm'))
        self.assertTrue(hasattr(res, 'warm_start'))
        self.assertTrue(hasattr(res, 'verbose'))
        self.assertTrue(hasattr(res, 'input_text_size'))
        self.assertTrue(hasattr(res, 'output_text_size'))
        self.assertTrue(hasattr(res, 'validation_fraction'))
        self.assertTrue(hasattr(res, 'tokenizer'))
        self.assertTrue(hasattr(res, 'input_text_size_'))
        self.assertTrue(hasattr(res, 'output_text_size_'))
        self.assertTrue(hasattr(res, 'input_vector_size_'))
        self.assertTrue(hasattr(res, 'output_vector_size_'))
        self.assertTrue(hasattr(res, 'vae_encoder_'))
        self.assertTrue(hasattr(res, 'vae_decoder_'))

    def test_fit_positive04(self):
        self.text_vae.verbose = 2
        print('')
        print('PRE-FITTING')
        res = self.text_vae.fit(self.target_texts)
        res.warm_start = True
        print('')
        print('FINE-TUNING')
        res.fit(self.target_texts, self.input_texts)
        self.assertIsInstance(res, Conv1dTextVAE)
        self.assertTrue(hasattr(res, 'n_filters'))
        self.assertTrue(hasattr(res, 'kernel_size'))
        self.assertTrue(hasattr(res, 'input_embeddings'))
        self.assertTrue(hasattr(res, 'output_embeddings'))
        self.assertTrue(hasattr(res, 'batch_size'))
        self.assertTrue(hasattr(res, 'max_epochs'))
        self.assertTrue(hasattr(res, 'latent_dim'))
        self.assertTrue(hasattr(res, 'use_batch_norm'))
        self.assertTrue(hasattr(res, 'max_dist_between_output_synonyms'))
        self.assertTrue(hasattr(res, 'warm_start'))
        self.assertTrue(hasattr(res, 'verbose'))
        self.assertTrue(hasattr(res, 'input_text_size'))
        self.assertTrue(hasattr(res, 'output_text_size'))
        self.assertTrue(hasattr(res, 'validation_fraction'))
        self.assertTrue(hasattr(res, 'tokenizer'))
        self.assertTrue(hasattr(res, 'input_text_size_'))
        self.assertTrue(hasattr(res, 'output_text_size_'))
        self.assertTrue(hasattr(res, 'input_vector_size_'))
        self.assertTrue(hasattr(res, 'output_vector_size_'))
        self.assertTrue(hasattr(res, 'vae_encoder_'))
        self.assertTrue(hasattr(res, 'vae_decoder_'))

    def test_fit_negative01(self):
        true_err_msg = re.escape('Length of `X` does not equal to length of `y`! {0} != {1}.'.format(
            len(self.input_texts), len(self.target_texts) - 1))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.text_vae.fit(self.input_texts, self.target_texts[:-1])

    def test_fit_negative02(self):
        X = copy.deepcopy(self.input_texts)
        X[1] = 2
        true_err_msg = re.escape('Item 1 of the parameter `X` is wrong! This item is not string!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.text_vae.fit(X, self.target_texts)

    def test_fit_negative03(self):
        self.text_vae.batch_size = -3
        true_err_msg = re.escape('The parameter `batch_size` is wrong! Expected a positive value, but -3 is not '
                                 'positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.text_vae.fit(self.input_texts, self.target_texts)

    def test_fit_negative04(self):
        del self.text_vae.input_embeddings
        true_err_msg = re.escape('The parameter `input_embeddings` is not defined!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.text_vae.fit(self.input_texts, self.target_texts)

    def test_fit_negative05(self):
        self.text_vae.input_embeddings = 4
        true_err_msg = re.escape('The parameter `input_embeddings` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(FastTextKeyedVectors(vector_size=300, min_n=1, max_n=5)), type(4)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.text_vae.fit(self.input_texts, self.target_texts)

    def test_fit_negative06(self):
        del self.text_vae.output_embeddings
        true_err_msg = re.escape('The parameter `output_embeddings` is not defined!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.text_vae.fit(self.input_texts, self.target_texts)

    def test_fit_negative07(self):
        self.text_vae.output_embeddings = 3.5
        true_err_msg = re.escape('The parameter `output_embeddings` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(FastTextKeyedVectors(vector_size=300, min_n=1, max_n=5)), type(3.5)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.text_vae.fit(self.input_texts, self.target_texts)

    def test_transform_positive01(self):
        res = self.text_vae.fit_transform(self.input_texts, self.target_texts)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape[0], len(self.input_texts))
        self.assertEqual(res.shape[1], self.text_vae.latent_dim)

    def test_transform_negative01(self):
        with self.assertRaises(NotFittedError):
            _ = self.text_vae.transform(self.input_texts)

    def test_transform_negative02(self):
        texts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        true_err_msg = re.escape('The parameter `X` is wrong! Expected 1-D array, got {0}-D array.'.format(
            len(texts.shape)))
        self.text_vae.fit(self.input_texts)
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = self.text_vae.transform(texts)

    def test_transform_negative03(self):
        self.text_vae.fit(self.input_texts)
        del self.text_vae.input_embeddings
        true_err_msg = re.escape('The parameter `input_embeddings` is not defined!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            self.text_vae.fit(self.input_texts, self.target_texts)

    def test_predict_positive01(self):
        batch_size = 3
        while (len(self.input_texts) % batch_size) == 0:
            batch_size += 1
        self.text_vae.batch_size = batch_size
        self.text_vae.max_epochs *= 2
        res = self.text_vae.fit_predict(self.input_texts + self.input_texts, self.target_texts + self.target_texts)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(res.shape, (2 * len(self.input_texts), self.text_vae.output_text_size_,
                                     self.text_vae.output_vector_size_))
        for sample_idx in range(res.shape[0]):
            for time_idx in range(res.shape[1]):
                self.assertAlmostEqual(np.linalg.norm(res[sample_idx][time_idx]), 1.0, places=4)

    def test_predict_negative01(self):
        with self.assertRaises(NotFittedError):
            _ = self.text_vae.predict(self.input_texts)

    def test_predict_negative02(self):
        texts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        true_err_msg = re.escape('The parameter `X` is wrong! Expected 1-D array, got {0}-D array.'.format(
            len(texts.shape)))
        self.text_vae.fit(self.input_texts)
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = self.text_vae.predict(texts)

    def test_predict_negative03(self):
        true_err_msg = re.escape('The parameter `input_embeddings` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(FastTextKeyedVectors(vector_size=300, min_n=1, max_n=5)), type(None)))
        self.text_vae.fit(self.input_texts)
        self.text_vae.input_embeddings = None
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = self.text_vae.predict(self.input_texts)

    def test_serialize_unfitted(self):
        with open(self.model_name, 'wb') as fp:
            pickle.dump(self.text_vae, fp)
        with open(self.model_name, 'rb') as fp:
            res = pickle.load(fp)
        self.assertIsInstance(res, Conv1dTextVAE)
        self.assertIsNot(res, self.text_vae)
        self.assertTrue(hasattr(res, 'n_filters'))
        self.assertTrue(hasattr(res, 'kernel_size'))
        self.assertTrue(hasattr(res, 'input_embeddings'))
        self.assertTrue(hasattr(res, 'output_embeddings'))
        self.assertTrue(hasattr(res, 'batch_size'))
        self.assertTrue(hasattr(res, 'max_epochs'))
        self.assertTrue(hasattr(res, 'latent_dim'))
        self.assertTrue(hasattr(res, 'max_dist_between_output_synonyms'))
        self.assertTrue(hasattr(res, 'use_batch_norm'))
        self.assertTrue(hasattr(res, 'warm_start'))
        self.assertTrue(hasattr(res, 'verbose'))
        self.assertTrue(hasattr(res, 'input_text_size'))
        self.assertTrue(hasattr(res, 'output_text_size'))
        self.assertTrue(hasattr(res, 'validation_fraction'))
        self.assertTrue(hasattr(res, 'tokenizer'))
        self.assertFalse(hasattr(res, 'input_text_size_'))
        self.assertFalse(hasattr(res, 'output_text_size_'))
        self.assertFalse(hasattr(res, 'input_vector_size_'))
        self.assertFalse(hasattr(res, 'output_vector_size_'))
        self.assertFalse(hasattr(res, 'vae_encoder_'))
        self.assertFalse(hasattr(res, 'vae_decoder_'))
        self.assertEqual(res.n_filters, self.text_vae.n_filters)
        self.assertEqual(res.kernel_size, self.text_vae.kernel_size)
        self.assertIsNone(res.input_embeddings)
        self.assertIsNone(res.output_embeddings)
        self.assertEqual(res.batch_size, self.text_vae.batch_size)
        self.assertEqual(res.use_batch_norm, self.text_vae.use_batch_norm)
        self.assertEqual(res.max_epochs, self.text_vae.max_epochs)
        self.assertEqual(res.latent_dim, self.text_vae.latent_dim)
        self.assertAlmostEqual(res.max_dist_between_output_synonyms, self.text_vae.max_dist_between_output_synonyms)
        self.assertEqual(res.warm_start, self.text_vae.warm_start)
        self.assertEqual(res.verbose, self.text_vae.verbose)
        self.assertEqual(res.input_text_size, self.text_vae.input_text_size)
        self.assertEqual(res.output_text_size, self.text_vae.output_text_size)
        self.assertEqual(res.validation_fraction, self.text_vae.validation_fraction)

    def test_serialize_fitted(self):
        self.text_vae.fit(self.input_texts, self.target_texts)
        with open(self.model_name, 'wb') as fp:
            pickle.dump(self.text_vae, fp)
        with open(self.model_name, 'rb') as fp:
            res = pickle.load(fp)
        self.assertIsInstance(res, Conv1dTextVAE)
        self.assertIsNot(res, self.text_vae)
        self.assertTrue(hasattr(res, 'n_filters'))
        self.assertTrue(hasattr(res, 'kernel_size'))
        self.assertTrue(hasattr(res, 'input_embeddings'))
        self.assertTrue(hasattr(res, 'output_embeddings'))
        self.assertTrue(hasattr(res, 'batch_size'))
        self.assertTrue(hasattr(res, 'max_epochs'))
        self.assertTrue(hasattr(res, 'latent_dim'))
        self.assertTrue(hasattr(res, 'max_dist_between_output_synonyms'))
        self.assertTrue(hasattr(res, 'use_batch_norm'))
        self.assertTrue(hasattr(res, 'warm_start'))
        self.assertTrue(hasattr(res, 'verbose'))
        self.assertTrue(hasattr(res, 'input_text_size'))
        self.assertTrue(hasattr(res, 'output_text_size'))
        self.assertTrue(hasattr(res, 'validation_fraction'))
        self.assertTrue(hasattr(res, 'tokenizer'))
        self.assertTrue(hasattr(res, 'input_text_size_'))
        self.assertTrue(hasattr(res, 'output_text_size_'))
        self.assertTrue(hasattr(res, 'input_vector_size_'))
        self.assertTrue(hasattr(res, 'output_vector_size_'))
        self.assertTrue(hasattr(res, 'vae_encoder_'))
        self.assertTrue(hasattr(res, 'vae_decoder_'))
        self.assertEqual(res.n_filters, self.text_vae.n_filters)
        self.assertEqual(res.kernel_size, self.text_vae.kernel_size)
        self.assertIsNone(res.input_embeddings)
        self.assertIsNone(res.output_embeddings)
        self.assertEqual(res.batch_size, self.text_vae.batch_size)
        self.assertEqual(res.max_epochs, self.text_vae.max_epochs)
        self.assertEqual(res.latent_dim, self.text_vae.latent_dim)
        self.assertAlmostEqual(res.max_dist_between_output_synonyms, self.text_vae.max_dist_between_output_synonyms)
        self.assertEqual(res.warm_start, self.text_vae.warm_start)
        self.assertEqual(res.verbose, self.text_vae.verbose)
        self.assertEqual(res.use_batch_norm, self.text_vae.use_batch_norm)
        self.assertEqual(res.input_text_size, self.text_vae.input_text_size)
        self.assertEqual(res.output_text_size, self.text_vae.output_text_size)
        self.assertEqual(res.validation_fraction, self.text_vae.validation_fraction)
        res.input_embeddings = self.text_vae.input_embeddings
        res.output_embeddings = self.text_vae.output_embeddings
        X1 = self.text_vae.transform(self.input_texts)
        X2 = res.transform(self.input_texts)
        self.assertIsInstance(X1, np.ndarray)
        self.assertIsInstance(X2, np.ndarray)
        self.assertEqual(X1.shape, X2.shape)
        self.assertEqual(X1.dtype, X2.dtype)

    def test_postprocess_reconstructed_word_vectors_positive01(self):
        source_word_vectors = np.hstack(
            (
                np.random.uniform(-1.0, 1.0, (20, 6)),
                np.random.uniform(-3.0, -2.0, (20, 1))
            )
        )
        true_end_idx = 11
        source_word_vectors[true_end_idx] = np.zeros((source_word_vectors.shape[1],), dtype=source_word_vectors.dtype)
        source_word_vectors[true_end_idx][source_word_vectors.shape[1] - 1] = 1.0
        true_word_vectors = np.zeros((source_word_vectors.shape[0], source_word_vectors.shape[1] - 1))
        for idx in range(source_word_vectors.shape[0]):
            if idx >= true_end_idx:
                break
            true_word_vectors[idx] = source_word_vectors[idx][:(source_word_vectors.shape[1] - 1)]
            true_word_vectors[idx] /= np.linalg.norm(true_word_vectors[idx])
        predicted_word_vectors = Conv1dTextVAE.postprocess_reconstructed_word_vectors(
            np.reshape(source_word_vectors, newshape=(1, source_word_vectors.shape[0], source_word_vectors.shape[1]))
        )
        self.assertIsInstance(predicted_word_vectors, np.ndarray)
        self.assertEqual(len(predicted_word_vectors.shape), 3)
        self.assertEqual(predicted_word_vectors.shape[0], 1)
        predicted_word_vectors = predicted_word_vectors[0]
        self.assertEqual(predicted_word_vectors.shape, true_word_vectors.shape)
        for word_idx in range(true_word_vectors.shape[0]):
            for feature_idx in range(true_word_vectors.shape[1]):
                self.assertAlmostEqual(
                    predicted_word_vectors[word_idx][feature_idx], true_word_vectors[word_idx][feature_idx],
                    msg='word_idx={0}, feature_idx={1}'.format(word_idx, feature_idx)
                )

    def test_postprocess_reconstructed_word_vectors_positive02(self):
        source_word_vectors = np.hstack(
            (
                np.random.uniform(-1.0, 1.0, (20, 6)),
                np.random.uniform(-3.0, -2.0, (20, 1))
            )
        )
        true_end_idx = 1
        source_word_vectors[true_end_idx] = np.zeros((source_word_vectors.shape[1],), dtype=source_word_vectors.dtype)
        source_word_vectors[true_end_idx][source_word_vectors.shape[1] - 1] = 1.0
        source_word_vectors[0][source_word_vectors.shape[1] - 1] = 1.5
        true_word_vectors = np.zeros((source_word_vectors.shape[0], source_word_vectors.shape[1] - 1))
        for idx in range(source_word_vectors.shape[0]):
            if idx >= true_end_idx:
                break
            true_word_vectors[idx] = source_word_vectors[idx][:(source_word_vectors.shape[1] - 1)]
            true_word_vectors[idx] /= np.linalg.norm(true_word_vectors[idx])
        predicted_word_vectors = Conv1dTextVAE.postprocess_reconstructed_word_vectors(
            np.reshape(source_word_vectors, newshape=(1, source_word_vectors.shape[0], source_word_vectors.shape[1]))
        )
        self.assertIsInstance(predicted_word_vectors, np.ndarray)
        self.assertEqual(len(predicted_word_vectors.shape), 3)
        self.assertEqual(predicted_word_vectors.shape[0], 1)
        predicted_word_vectors = predicted_word_vectors[0]
        self.assertEqual(predicted_word_vectors.shape, true_word_vectors.shape)
        for word_idx in range(true_word_vectors.shape[0]):
            for feature_idx in range(true_word_vectors.shape[1]):
                self.assertAlmostEqual(
                    predicted_word_vectors[word_idx][feature_idx], true_word_vectors[word_idx][feature_idx],
                    msg='word_idx={0}, feature_idx={1}'.format(word_idx, feature_idx)
                )

    def test_postprocess_reconstructed_word_vectors_negative01(self):
        source_word_vectors = np.hstack(
            (
                np.random.uniform(-1.0, 1.0, (20, 6)),
                np.random.uniform(-3.0, -2.0, (20, 1))
            )
        )
        true_end_idx = 11
        source_word_vectors[true_end_idx] = np.zeros((source_word_vectors.shape[1],), dtype=source_word_vectors.dtype)
        source_word_vectors[true_end_idx][source_word_vectors.shape[1] - 1] = 1.0
        true_err_msg = re.escape('The `word_vectors` parameter is wrong! Expected 3-D array, got 2-D array!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _ = Conv1dTextVAE.postprocess_reconstructed_word_vectors(source_word_vectors)


if __name__ == '__main__':
    unittest.main(verbosity=2)
