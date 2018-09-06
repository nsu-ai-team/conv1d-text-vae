import gzip
import os
from requests import get
import shutil
import tarfile

from gensim.models import FastText
from gensim.models.keyedvectors import FastTextKeyedVectors


def load_russian_fasttext_rusvectores() -> FastTextKeyedVectors:
    fasttext_ru_dirname = os.path.join(os.path.dirname(__file__), '..', 'data',
                                       'araneum_none_fasttextcbow_300_5_2018')
    if not os.path.isdir(fasttext_ru_dirname):
        os.mkdir(fasttext_ru_dirname)
        url = 'http://rusvectores.org/static/models/rusvectores4/fasttext/araneum_none_fasttextcbow_300_5_2018.tgz'
        fasttext_ru_archive = os.path.join(os.path.dirname(__file__), '..', 'data',
                                           'araneum_none_fasttextcbow_300_5_2018.tgz')
        try:
            with open(fasttext_ru_archive, 'wb') as fp:
                response = get(url)
                fp.write(response.content)
            with tarfile.open(fasttext_ru_archive, 'r:gz') as fasttext_tar:
                fasttext_tar.extractall(fasttext_ru_dirname)
        finally:
            os.remove(fasttext_ru_archive)
    return FastText.load(os.path.join(fasttext_ru_dirname, 'araneum_none_fasttextcbow_300_5_2018.model')).wv


def load_russian_fasttext() -> FastTextKeyedVectors:
    fasttext_ru_modelname = os.path.join(os.path.dirname(__file__), '..', 'data', 'cc.ru.300.bin')
    if not os.path.isfile(fasttext_ru_modelname):
        url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.ru.300.bin.gz'
        fasttext_ru_archive = os.path.join(os.path.dirname(__file__), '..', 'data', 'cc.ru.300.bin.gz')
        try:
            with open(fasttext_ru_archive, 'wb') as fp:
                response = get(url)
                fp.write(response.content)
            with gzip.open(fasttext_ru_archive, 'rb') as fasttext_gzip, open(fasttext_ru_modelname, 'wb') as res_fp:
                shutil.copyfileobj(fasttext_gzip, res_fp, length=1024)
        finally:
            os.remove(fasttext_ru_archive)
    return FastText.load_fasttext_format(fasttext_ru_modelname).wv


def load_english_fasttext() -> FastTextKeyedVectors:
    fasttext_en_modelname = os.path.join(os.path.dirname(__file__), '..', 'data', 'cc.en.300.bin')
    if not os.path.isfile(fasttext_en_modelname):
        url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/cc.en.300.bin.gz'
        fasttext_en_archive = os.path.join(os.path.dirname(__file__), '..', 'data', 'cc.en.300.bin.gz')
        try:
            with open(fasttext_en_archive, 'wb') as fp:
                response = get(url)
                fp.write(response.content)
            with gzip.open(fasttext_en_archive, 'rb') as fasttext_gzip, open(fasttext_en_modelname, 'wb') as res_fp:
                shutil.copyfileobj(fasttext_gzip, res_fp, length=1024)
        finally:
            os.remove(fasttext_en_archive)
    return FastText.load_fasttext_format(fasttext_en_modelname).wv
