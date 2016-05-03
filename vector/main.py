__author__ = 'HyNguyen'

import time
import numpy as np
from wordvectors import WordVectors
import logging
from gensim.models import word2vec


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_glove( embsize ,filename):
    word_index = {}
    with open(filename, mode="r") as f:
        for i,line in enumerate(f):
            elements = line.split()
            if i % 10000 == 0:
                print("process line", i)
            word = elements[0]
            if not word.islower() and not word.isdigit():
                print word

if __name__ == "__main__":

    a = np.array(("1","2"),dtype=float)
    print a









