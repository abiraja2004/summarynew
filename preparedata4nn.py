__author__ = 'HyNguyen'
import os
import pickle
import numpy as np
import xml.etree.ElementTree as ET
from vector.wordvectors import WordVectors

class RougeScore(object):
    def __init__(self, rouge_1=0, rouge_2=0, rouge_su4=0):
        self.rouge_1 = rouge_1
        self.rouge_2 = rouge_2
        self.rouge_su4 = rouge_su4

    def add_score(self, name, score):
        if name == "ROUGE-1":
            self.rouge_1 = score
        elif name == "ROUGE-2":
            self.rouge_2 = score
        elif name == "ROUGE-SU4":
            self.rouge_su4 = score

def generate_data(name):
    """
    Params
        name: name of data
    Returns
        rouge_data: [n_sample x 3]
    """
    rouge_data = {}
    with open("data/"+name+".1.out.txt", mode="r") as f:
        for line in f:
            elements = line.split()
            if len(elements) == 7:
                _, rouge_name, _, item_id, R, P, F = elements
                id = item_id[:-2]
                if id not in rouge_data.keys():
                    rouge_data[id] = RougeScore()
                rouge_data[id].add_score(rouge_name, float(F[2:]))
    print(len(rouge_data.keys()))
    return rouge_data

def generate_id_score():
    """
    generate rouge score data [n_sample x 3], dailymail, duc04, duc05 and Save to file picke (dailymail, duc04, duc05)
    """
    dailymail = generate_data("dailymail")
    duc04 = generate_data("duc04")
    duc05 = generate_data("duc05")
    with open("data/score.dailymail.duc04.duc05.pickle", mode="wb") as f:
        pickle.dump((dailymail,duc04,duc05), f)

def load_id_score():
    """
    load data (dailymail, duc04, duc05) from file
    """
    if not os.path.isfile("data/score.dailymail.duc04.duc05.pickle"):
        generate_id_score()
    with open("data/score.dailymail.duc04.duc05.pickle", mode="rb") as f:
        dailymail, duc04, duc05 = pickle.load(f)
    return (dailymail, duc04, duc05)


def prepare_sentence_score(id_score, file_xml, file_out, name = ""):
    """
    Params
        id_score: rouge score with id of ROUGE1.5.5, id is key in dic{}
        file_xml: file setting_<name>.xml
        file_out: file save data
        name: name of data
    """
    file_path = file_xml
    tree = ET.parse(file_path)
    root = tree.getroot()

    fo = open(file_out, mode="w")

    if not os.path.exists("data/" + name):
        os.makedirs("data/" + name)

    for child in root._children:
        id = child.attrib["ID"]
        path_peer = child._children[1].text.replace("\n","").replace("\t","")
        path_peer_file = child._children[3]._children[0].text
        if name == "duc04":
            print("ttdt")
        with open("data/" + path_peer + "/" + path_peer_file, mode="r") as f:
            sentence = f.read().replace("\n"," ")
        fo.write(sentence + " hynguyensplit " + str(id_score[id].rouge_1) + " " + str(id_score[id].rouge_2) + " " + str(id_score[id].rouge_su4) + "\n")
    fo.close()

def prepare_sentence_score_all_data():
    """
    Create "sentence hynguyensplit rouge_score1 rouge_score2 rouge_scoresu4" and write to file
    """
    dailymail, duc04, duc05 = load_id_score()
    print(len(dailymail), len(duc04), len(duc05))
    prepare_sentence_score(duc04, "data/setting_duc04.xml", "data/sentence.score.duc04.txt", name="duc04")
    prepare_sentence_score(duc05, "data/setting_duc05.xml", "data/sentence.score.duc05.txt", name="duc05")
    prepare_sentence_score(dailymail, "data/setting_dailymail.xml", "data/sentence.score.dailymail.txt", name="dailymail")


# 1: word2vec
# 2: glove
# 3: cwvector
# -- read data and get wordvector base on data

# 4: random -- vocabulary of 1|2|3 vector random

def prepare_indexs_score_of_file(filename, wordvector, data_name ,mincount = 10, maxcount = 70):
    indexs_final = []
    rouge_score_final = []
    with open(filename, mode="r") as f:
        for i,line in enumerate(f):
            if i % 1000 == 0:
                print("convert index line ", i)
            sentence, score = line.split("hynguyensplit")
            sentence_length = sentence.count(" ")
            if sentence_length >=mincount and sentence_length <=maxcount:
                indexs = wordvector.get_string_index(sentence)
                sample = [0]*5 + indexs + [0]*(maxcount +10 - 5 -len(indexs))
                if len(sample) != 80:
                    print sentence
                    continue
                indexs_final.append(np.array(sample,dtype=np.int32))
                rouge_score_final.append(np.array(score.split(), dtype=np.float32))

    indexs_final = np.array(indexs_final,dtype=np.int32)
    rouge_score_final = np.array(rouge_score_final, dtype=np.float32)

    print (indexs_final.shape, rouge_score_final.shape)
    if not os.path.exists("data/"+wordvector.name):
        os.makedirs("data/"+wordvector.name)
    np.save("data/"+wordvector.name+"/index."+data_name, indexs_final)
    np.save("data/"+wordvector.name+"/score."+data_name, rouge_score_final)
    return indexs_final, rouge_score_final

def staticstic_data_file(filename):
    histo = [0]*200
    with open(filename, mode="r") as f:
        for line in f:
            sentence, score = line.split("hynguyensplit")
            elements = sentence.split()
            if len(elements) < 1:
                print(elements)
            idx = sentence.count(" ")
            try:
                histo[idx] += 1
                if idx > 70 and idx < 100:
                    print sentence
            except:
                pass

    for khung in histo:
        print khung

if __name__ == "__main__":

    wordvector_w2v = WordVectors.load_from_text_format("model/word2vec.txt", name="word2vec")
    prepare_indexs_score_of_file("data/sentence.score.dailymail.txt", wordvector_w2v, "dailymail")
    prepare_indexs_score_of_file("data/sentence.score.duc04.txt", wordvector_w2v, "duc04")
    prepare_indexs_score_of_file("data/sentence.score.duc05.txt", wordvector_w2v, "duc05")

    wordvector_glove = WordVectors.load_from_text_format("model/glove.filter.txt", name="glove")
    prepare_indexs_score_of_file("data/sentence.score.dailymail.txt", wordvector_glove, "dailymail")
    prepare_indexs_score_of_file("data/sentence.score.duc04.txt", wordvector_glove, "duc04")
    prepare_indexs_score_of_file("data/sentence.score.duc05.txt", wordvector_glove, "duc05")

    wordvector_cw = WordVectors.load_from_text_format("model/cwvector.txt", name="cw")
    prepare_indexs_score_of_file("data/sentence.score.dailymail.txt", wordvector_cw, "dailymail")
    prepare_indexs_score_of_file("data/sentence.score.duc04.txt", wordvector_cw, "duc04")
    prepare_indexs_score_of_file("data/sentence.score.duc05.txt", wordvector_cw, "duc05")



