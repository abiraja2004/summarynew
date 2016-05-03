__author__ = 'HyNguyen'

import pickle
from gensim.models import word2vec
from wordvectors import WordVectors

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    # Load Word2Vec from Google
    w2v = word2vec.Word2Vec.load_word2vec_format("/Users/HyNguyen/Documents/Research/Data/GoogleNews-vectors-negative300.bin",binary=True)
    with open("vocab.notlower.pickle", mode="rb") as f:
        vocab = pickle.load(f)
    wordvectors = WordVectors.create_wordvectos_from_word2vec_vocab(w2v,vocab)
    wordvectors.save_text_format("../model/word2vec.txt")