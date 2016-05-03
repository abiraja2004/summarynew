__author__ = 'HyNguyen'

import numpy as np

from vector.wordvectors import WordVectors
import sys
import time
from nltk.parse.stanford import StanfordParser

if __name__ == "__main__":
    a = WordVectors.load_from_text_format("model/word2vec.txt",name= "word2vec")
    np.save("data/word2vec/embed_matrix.npy",a.embed_matrix)


