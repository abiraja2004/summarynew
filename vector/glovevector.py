__author__ = 'HyNguyen'

import os
import numpy as np
from wordvectors import WordVectors
from nltk.corpus import treebank
from nltk.corpus import brown
import nltk
import xml.etree.ElementTree as ET
from gensim.models import word2vec
import os

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_word_from_data():
    vocab = {}
    # Penn Tree Bank
    treebank_sents = treebank.sents()
    for i in range(len(treebank_sents)):
        for word in treebank_sents[i]:
            vocab[str(word).lower()] = 1
    print("Finish Penn Tree Bank corpus, vocab size: ", str(len(vocab.keys())))

    # Brown
    brown_sents = brown.sents()
    for i in range(len(brown_sents)):
        for word in brown_sents[i]:
            vocab[str(word).lower()] = 1
    print("Finish Broww corpus, vocab size: ", str(len(vocab.keys())))

    # dailymail data
    with open("../data/sentence.score.dailymail.txt", mode="r") as f:
        for line in f:
            sentence, score = line.split("hynguyensplit")
            words = nltk.word_tokenize(sentence)
            for word in words:
                vocab[str(word).lower()] = 1

    # duc04 data
    with open("../data/sentence.score.duc04.txt", mode="r") as f:
        for line in f:
            sentence, score = line.split("hynguyensplit")
            words = nltk.word_tokenize(sentence)
            for word in words:
                vocab[str(word).lower()] = 1

    # duc05 data
    with open("../data/sentence.score.duc05.txt", mode="r") as f:
        for line in f:
            sentence, score = line.split("hynguyensplit")
            words = nltk.word_tokenize(sentence)
            for word in words:
                vocab[str(word).lower()] = 1

    print("Finish reading vocab size: ", str(len(vocab.keys())))
    return vocab

import pickle
if __name__ == "__main__":

    w2v = word2vec.Word2Vec.load_word2vec_format("/Users/HyNguyen/Documents/MachineLearning/convae/model/glove.400k.txt",binary=False)
    with open("vocab.lower.pickle", mode="rb") as f:
        vocab = pickle.load(f)
    wordvectors = WordVectors.create_wordvectos_from_word2vec_vocab(w2v,vocab)
    wordvectors.save_text_format("../model/glove.filter.txt")