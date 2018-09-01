from setuptools import setup, find_packages

import seq2seq_lstm

long_description = '''
conv1d-text-vae
============
The Conv1D-Text-VAE is a special convolutional VAE (variational autoencoder)
for the text generation and the text semantic hashing. This package with the
sklearn-like interface, and it uses the Keras package for neural modeling and
the Fasttext models for "online" word vectorizing.
Getting Started
---------------
Installing
~~~~~~~~~~
To install this project on your local machine, you should run the
following commands in Terminal:
.. code::
    git clone https://github.com/nsu-ai/conv1d-text-vae.git
    cd conv1d-text-vae
    sudo python setup.py
You can also run the tests:
.. code::
    python setup.py test
'''

setup(
    name='seq2seq-lstm',
    version=seq2seq_lstm.__version__,
    packages=find_packages(exclude=['tests', 'demo']),
    include_package_data=True,
    description='A convolutional variational autoencoder for text generation and semantic hashing '
                'with the simple sklearn-like interface',
    long_description=long_description,
    url='https://github.com/nsu-ai/conv1d-text-vae',
    author='Ivan Bondarenko',
    author_email='bond005@yandex.ru',
    license='Apache License Version 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords=['vae', 'conv1d', 'nlp', 'keras', 'scikit-learn'],
    install_requires=['gensim>=3.5.0', 'h5py>=2.8.0', 'Keras>=2.2.0', 'numpy>=1.14.5', 'scipy>=1.1.0', 'nltk>=3.2.5',
                      'scikit-learn>=0.19.1', 'requests>=2.19.1'],
    test_suite='tests'
)