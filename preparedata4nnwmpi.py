from vector.wordvectors import WordVectors
import time
import numpy as np

from nltk.corpus import brown
from nltk.corpus import treebank
import nltk
import xml.etree.ElementTree as ET
import os

import sys

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if __name__ == "__main__":
    data_scatters = []
    start_total = 0
    max_word = 70
    min_word = 10
    if rank == 0:
        start_total = time.time()
        wordvectors = WordVectors.load_from_text_format("model/cwvector.txt", name="word2vec")
        print("Finished read wordvectors ...")
        with open("data/sentence.score.duc04.txt", mode="r") as f:
            traindata = f.readlines()
        size_sample = int(len(traindata)/size)
        for i in range(size):
            if i* size_sample + size_sample > len(traindata):
                data_scatters.append(traindata[i*size_sample:])
            else:
                data_scatters.append(traindata[i*size_sample : i*size_sample+size_sample])
    else:
        wordvectors = None
        data_scatter = None

    wordvectors = comm.bcast(wordvectors, root = 0)
    print("Process:", rank, "broadcasted wordvectors ...")
    data_scatter = comm.scatter(data_scatters,root=0)
    print("Process:", rank, "Data scatter length: ", len(data_scatter))

    #work with data_scatter
    indexs_final = []
    rouge_score_final = []
    for i,line in enumerate(data_scatter):
        if i % 1000 == 0:
            print("Process:", rank, "convert index line ", i)
        sentence, score = line.split("hynguyensplit")
        sentence_length = sentence.count(" ")
        if sentence_length >= min_word and sentence_length <=max_word:
            indexs = wordvectors.get_string_index(sentence)
            sample = [0]*5 + indexs + [0]*(max_word +10 - 5 -len(indexs))
            indexs_final.append(np.array(sample,dtype=np.int32))
            rouge_score_final.append(np.array(score.split(), dtype=np.float32))

    indexs_final = np.array(indexs_final)
    rouge_score_final = np.array(rouge_score_final)
    print("Process:", rank, "Data final array shape: ", indexs_final.shape)
    print("Process:", rank, "Data score array shape: ", rouge_score_final.shape)

    data_index_gather = comm.gather(indexs_final, root=0)
    data_score_gather = comm.gather(rouge_score_final, root =0)

    if rank == 0:
        # gather and save
        print("data gather")
        data_index_final = data_index_gather[0]
        data_score_final = data_score_gather[0]

        for i in range(1,len(data_index_gather)):
            data_index_final = np.concatenate((data_index_final,data_index_gather[i]))
            data_score_final = np.concatenate((data_score_final,data_score_gather[i]))

        print("Process:", rank, "data_index_final.shape: ", data_index_final.shape)
        print("Process:", rank, "data_score_final.shape: ", data_score_final.shape)
        end_total = time.time()
        print("Process:", rank, "Total time: ", end_total - start_total, "s")
        np.save("data/data.4nn.index", data_index_final)
        np.save("data/data.4nn.score", data_score_final)
        print("Process:", rank, "Save to data/data.convae.np ")