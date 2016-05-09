__author__ = 'HyNguyen'
from nltk.parse.stanford import StanfordParser
from nltk.treetransforms import chomsky_normal_form
from nltk.tree import Tree
from vector.wordvectors import WordVectors
parser = StanfordParser(path_to_jar="/Users/HyNguyen/Downloads/stanford-parser-full-2015-12-09/stanford-parser.jar",
                        path_to_models_jar="/Users/HyNguyen/Downloads/stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar"
                        ,model_path="/Users/HyNguyen/Downloads/stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")


import pickle


def prepapre_sentiment_data():
    with open("../data/rt-polarity.neg.txt",mode="r") as f:
        neg_sent = f.readlines()

    with open("../data/rt-polarity.pos.txt",mode="r") as f:
        pos_sent = f.readlines()

    sents_label = [0]*len(neg_sent) + [1]*len(pos_sent)

    trees = []
    labels = []

    count = 0
    for sent, label in zip(neg_sent+pos_sent, sents_label):
        try:
            count +=1
            if count % 200 ==0 :
                print(count)
            a = list(parser.raw_parse(sent))
            hytree = a[0]
            chomsky_normal_form(hytree)
            trees.append(hytree[0])
            labels.append(label)
        except:
            continue

    with open("hynguyen.pickle",mode="wb") as f:
        pickle.dump((trees,labels),f)


import numpy as np

if __name__ =="__main__":
    rng = np.random.RandomState(0)
    a = range(0,10)
    print a
    rng.shuffle(a)
    print a
    b = range(0,10)
    print b
    rng.shuffle(b)
    print b





