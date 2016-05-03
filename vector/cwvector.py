__author__ = 'HyNguyen'

import os
from wordvectors import WordVectors
import numpy as np

def read_from_senna():
    file_embedding  = "/Users/HyNguyen/Documents/Research/Data/senna/embeddings/embeddings.txt"
    file_wordslist  = "/Users/HyNguyen/Documents/Research/Data/senna/hash/words.lst"
    with open(file_wordslist, mode="r") as f:
        words = f.readlines()
    with open(file_embedding, mode="r") as f:
        vectors = f.readlines()
    words_embedding = []
    words_index = {}
    for i ,(word, vector) in enumerate(zip(words,vectors)):
        word_2 = word[:-1]
        vector_2 = vector[:-1].split()
        vec_np = np.array(vector_2,dtype=np.float32)
        words_embedding.append(vec_np)
        words_index[word_2] = i

    words_embedding = np.array(words_embedding,dtype=np.float32)
    print("words_embedding.shape",words_embedding.shape)
    print("words_index.length",len(words_index))
    wordvectors = WordVectors(50,words_embedding,words_index)
    wordvectors.save_text_format("../model/cwvector.txt")

if __name__ == "__main__":
    a = WordVectors.load_from_text_format("../model/cwvector.txt", "CWVector")



