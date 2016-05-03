__author__ = 'HyNguyen'
import numpy as np
import time
from gensim.models import word2vec
import pickle
import copy
import os
import nltk

class WordVectors(object):
    def __init__(self, embsize, embed_matrix, word_index, name = "word2vec"):
        self.embsize = embsize
        self.embed_matrix = embed_matrix
        self.word_index = word_index
        self.count_null_word = 0
        self.count_exist_word = 0
        self.name = name

    def add_wordvector_from_w2vmodel(self, word2vec, words):
        for word in words:
            try:
                vector = word2vec[word]
                if word in self.word_index.keys():
                    continue
                else:
                    self.word_index[word] = len(self.word_index.keys())
                    self.embed_matrix = np.concatenate((self.embed_matrix,vector.reshape(1,300)))
                    # print("hy")
                    # print(self.embed_matrix.shape)
                self.count_exist_word +=1
            except:
                self.count_null_word +=1
                continue

    @classmethod
    def create_wordvectos_from_word2vec_vocab(cls, word2vec, vocab):
        word_embedding = []
        word_index = {}
        count_exist_word = 0
        count_null_word = 0
        for word in vocab.keys():
            try:
                vector = word2vec[word]
                count_exist_word +=1
                if word in word_index.keys():
                    continue
                else:
                    word_index[word] = len(word_index.keys())
                    word_embedding.append(vector)
            except:
                count_null_word +=1
                continue
        word_embedding = np.array(word_embedding, dtype=np.float32)
        print("word_embedding.shape",word_embedding.shape)
        print("len(word_index.keys())",len(word_index.keys()))
        return WordVectors(word_embedding.shape[1],word_embedding,word_index)

    def save_text_format(self, filename):
        with open(filename, mode= "w") as f:
            if self.embed_matrix.shape[0] != len(self.word_index.keys()):
                print("co gi do sai sai")
            f.write(str(self.embed_matrix.shape[0]) + " " + str(self.embsize)+ "\n")
            print(self.embed_matrix.shape)
            for key in self.word_index.keys():
                index  = self.word_index[key]
                vector = self.embed_matrix[index].reshape(self.embsize)
                listnum = map(str, vector.tolist())
                f.write(key + " " + " ".join(listnum) + "\n")

    @classmethod
    def load_from_text_format(cls, filename, name):
        fi = open(filename,mode="r")
        dict_size, embsize = fi.readline().split()
        dict_size, embsize = int(dict_size), int(embsize)
        word_index = {"vector_0": 0, "UNK":1}
        counter = 2
        embed_matrix = [np.zeros(embsize,dtype=np.float32)]*2
        for i in range(1,dict_size+1,1):
            counter +=1
            if counter % 10000 == 0:
                print("Process wordvector line: ", counter)
            elements = fi.readline().split()
            word = elements[0]
            vector = np.array(elements[1:],dtype=np.float32)
            embed_matrix.append(vector)
            word_index[word] = i
        fi.close()
        embed_matrix = np.array(embed_matrix,dtype=np.float32)
        embed_matrix[1] = np.mean(embed_matrix[1:],axis=0,dtype=np.float32)
        return WordVectors(embsize,embed_matrix,word_index,name)

    @classmethod
    def load_glove(cls, embsize ,filename):
        word_index = {}
        embed_matrix = []
        with open(filename, mode="r") as f:
            for i,line in enumerate(f):
                elements = line.split()
                if i % 10000 == 0:
                    print("process line", i)
                word = elements[0]
                vector = np.array(elements[1:],dtype=np.float32)
                word_index[word] = i
                embed_matrix.append(vector)
            embed_matrix = np.array(embed_matrix,dtype=np.float32)
            embed_matrix[0] = np.mean(embed_matrix[1:],axis=0,dtype=np.float32)
            print("word_embedding.shape",embed_matrix.shape)
            print("len(word_index.keys())",len(word_index.keys()))
        return WordVectors(embsize,embed_matrix,word_index)

    def filter_wordvectors_vocab(self, vocab):
        embed_matrix = []
        word_index = {}
        count_exist_word = 0
        count_null_word = 0
        for i,word in enumerate(vocab.keys()):
            if i % 1000 == 0:
                print("filter word number,", i)
            if word in self.word_index.keys():
                count_exist_word +=1
                vector = self.wordvector(word)
                embed_matrix.append(vector)
                word_index[word] = i
            else:
                count_null_word +=1
        embed_matrix = np.array(embed_matrix, dtype=np.float32)
        print("word_embedding.shape",embed_matrix.shape)
        print("len(word_index.keys())",len(word_index.keys()))
        return WordVectors(embed_matrix.shape[1],embed_matrix,word_index)

    def get_word_index(self, word):
        if self.name != "word2vec":
            word = word.lower()
        if word in self.word_index.keys():
            return self.word_index[word]
        else:
            return 1

    def get_string_index(self, string):
        return [self.get_word_index(word) for word in nltk.word_tokenize(string)]

    def wordvector(self, word):
        if self.name != "word2vec":
            word = word.lower()
        if word in self.word_index.keys():
            self.count_exist_word +=1
            # print("try")
            return self.embed_matrix[self.word_index[word]]
        else:
            # print("except")
            #Null word
            self.count_null_word +=1
            return self.embed_matrix[1]

    def get_vector_addtion(self, words):
        result_vec = copy.deepcopy(self.wordvector(words[0]))
        for i in range(1,len(words)):
            result_vec += self.wordvector(words[i])
        return result_vec

    def prepare_index_from_string(self, sentence, min_length= 10, max_length=70):
        print "ttdtilu"


    def cae_prepare_data_from_string(self, sentence, min_length=10,  max_length=100):
        sentence = sentence.replace("\n","")
        elements = sentence.split()
        sentence_matrix = np.array([self.wordvector(word) for word in elements])
        padding = np.zeros((5,self.embsize),dtype=float)
        if sentence_matrix.shape[0] < max_length and sentence_matrix.shape[0] > min_length:
            sentence_matrix = np.concatenate((sentence_matrix,np.zeros((max_length-sentence_matrix.shape[0],self.embsize))))
        else:
            print(sentence)
            return None
        sentence_matrix_final = np.concatenate((padding,sentence_matrix,padding))
        return sentence_matrix_final

    def cae_prepare_data_from_words(self, words, min_length=10, max_length=100):
        sentence_matrix = np.array([self.wordvector(word) for word in words])
        padding = np.zeros((5,self.embsize),dtype=np.float32)
        if sentence_matrix.shape[0] <= max_length and sentence_matrix.shape[0] >= min_length:
            sentence_matrix = np.concatenate((sentence_matrix,np.zeros((max_length-sentence_matrix.shape[0],self.embsize))))
        else:
            # print(" ".join(words))
            return None
        sentence_matrix_final = np.concatenate((padding,sentence_matrix,padding))
        return sentence_matrix_final

if __name__ == "__main__":

    wordvector = WordVectors.load_from_text_format("../model/word2vec.txt","word2vec")
    print(wordvector.embed_matrix.shape)
    print(len(wordvector.word_index))




