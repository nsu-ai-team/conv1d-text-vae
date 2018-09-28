import unittest

import numpy as np

from conv1d_text_vae.conv1d_text_vae import Conv1dTextVAE, SequenceForVAE
from conv1d_text_vae.tokenizer import DefaultTokenizer
from conv1d_text_vae.fasttext_loading import load_russian_fasttext_rusvectores


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


if __name__ == '__main__':
    unittest.main(verbosity=2)
