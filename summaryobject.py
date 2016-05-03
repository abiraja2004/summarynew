__author__ = 'HyNguyen'
import os
import codecs
import numpy as np
import codecs
import xml.etree.ElementTree as ET
import nltk

class Cluster(object):
    def __init__(self, cluster_id ,list_documents, list_references):
        self.list_documents = list_documents
        self.list_references = list_references
        self.length_documents = len(list_documents)
        self.length_references = len(list_references)
        self.cluster_id = cluster_id
        self.my_summarys = []

    def get_average_length_ref(self):
        tmp = np.array([document.word_count for document in self.list_references])
        average = np.max(tmp)
        return average

    @classmethod
    def load_from_folder_vietnamese_mds(cls, cluster_id , cluster_path):
        if os.path.exists(cluster_path):
            files_name = os.listdir(cluster_path)
            list_documents = []
            list_references = []
            for file_name in files_name:
                file_prefix = file_name.find('.body.tok.txt')
                sentences = []
                document_id = ""
                if file_prefix > 0 :
                    document_id = file_name[:file_prefix]
                    file = codecs.open(cluster_path + '/' + file_name)
                    for line in file.readlines():
                        # remove name of authors
                        if len(line) < 50:
                            continue
                        sentences.append(Sentence(line))
                    list_documents.append(Document(sentences,document_id))
                    file.close()
                elif file_name.find(".ref") != -1 and file_name.find(".tok.txt") != -1:
                    fi = codecs.open(cluster_path + '/' + file_name)
                    lines = fi.readlines()
                    sentences = [Sentence(line,None) for line in lines]
                    fi.close()
                    document_id = "ref"
                    list_references.append(Document(sentences,document_id))
            return Cluster(cluster_id,list_documents, list_references)
        else:
            return None

    @classmethod
    def load_from_folder_duc05(cls, cluster_id, cluster_path, reference_path = ""):
        if os.path.exists(cluster_path):
            file_names = os.listdir(cluster_path)
            list_documents = []
            list_references = []
            for file_name in file_names:
                if file_name[0] == ".":
                    continue
                sentences_object = []
                file_path = cluster_path + "/" + file_name
                tree = ET.parse(file_path)
                root = tree.getroot()

                for child in root._children:
                    if child.tag == "TEXT":
                        text = child.text.replace("\n", " ")
                sentences = nltk.tokenize.sent_tokenize(text)
                for sentence in sentences:
                    sentences_object.append(Sentence(sentence))
                document_id = file_name
                list_documents.append(Document(sentences_object,document_id))

            all_ref_file = os.listdir(reference_path)
            file_prefix = cluster_id[:-1].upper()
            files_ref = [file_ref for file_ref in all_ref_file if file_ref.find(file_prefix) != -1]
            for file_ref in files_ref:
                list_references.append(Document.create_from_file(file_ref,reference_path+ "/" + file_ref))
            return Cluster(file_prefix, list_documents,list_references)
        else:
            print("Not a path")
            return None

    @classmethod
    def load_from_folder_duc04(cls, cluster_id, cluster_path, reference_path = ""):
        if os.path.exists(cluster_path):
            file_names = os.listdir(cluster_path)
            list_documents = []
            list_references = []
            for file_name in file_names:
                if file_name[0] == ".":
                    continue
                sentences_object = []
                file_path = cluster_path + "/" + file_name
                tree = ET.parse(file_path)
                root = tree.getroot()
                text_tag = root._children[3]
                if text_tag.tag == "TEXT":
                    text = text_tag.text.replace("\n", " ")
                sentences = nltk.tokenize.sent_tokenize(text)
                for sentence in sentences:
                    sentences_object.append(Sentence(sentence))
                document_id = file_name
                list_documents.append(Document(sentences_object,document_id))

            all_ref_file = os.listdir(reference_path)
            file_prefix = cluster_id[:-1].upper()
            files_ref = [file_ref for file_ref in all_ref_file if file_ref.find(file_prefix) != -1]
            for file_ref in files_ref:
                list_references.append(Document.create_from_file(file_ref,reference_path+ "/" + file_ref))
            return Cluster(file_prefix, list_documents,list_references)
        else:
            print("Not a path")
            return None

    def gen_data_for_rouge_method(self, model_path, peer_path):
        for reference in self.list_references:
            ref_sentences = reference.gen_sentences_str()
            with open(model_path + "/" +reference.document_id, mode="w") as f:
                f.writelines(ref_sentences)
        doc_sentences = self.gen_setntence_str()
        for i,sentence in enumerate(doc_sentences):
            with open(peer_path + "/" + self.cluster_id + "."+str(i), mode="w") as f:
                f.write(sentence)

    def gen_setntence_str(self):
        doc_sentences = []
        for document in self.list_documents:
            doc_sentences += document.gen_sentences_str()
        return doc_sentences

    @classmethod
    def load_from_opinosis(cls, cluster_id, cluster_path ,  wordvectors , reference_path = ""):
        # load data
        list_documents = []
        if os.path.exists(cluster_path):

            list_references = []
            sentences_object = []
            with open(cluster_path, mode="r") as f:
                sentences = f.readlines()
                for sentence in sentences:
                    words = nltk.word_tokenize(sentence)
                    sent_vec = wordvectors.get_vector_addtion(words)
                    sentences_object.append(Sentence(sentence,sent_vec))
                list_documents.append(Document(sentences_object,cluster_id))
        else:
            print("Not a path")
            return None

        list_references = []
        if os.path.exists(reference_path):
            # load reference
            for file_name in os.listdir(reference_path):
                sentences_object = []
                if file_name[0] == ".":
                    continue
                with open(reference_path+ "/" +file_name, mode="r") as f:
                    sentences = f.readlines()
                    for sentence in sentences:
                        sentences_object.append(Sentence(sentence))
                    list_references.append(Document(sentences_object,file_name))
        else:
            print("Not a path")
            return None

        return Cluster(cluster_id, list_documents,list_references)

class Document(object):
    def __init__(self,  list_sentences , document_id = -1,):
        self.list_sentences = list_sentences
        self.document_id = document_id
        self.length = len(list_sentences)
        self.word_count = sum([sentence.length for sentence in list_sentences if isinstance(sentence, Sentence)])

    @classmethod
    def create_from_file(cls, file_id ,file_path):
        list_sentences = []
        with open(file_path, mode="r") as f:
            lines = f.readlines()
            for line in lines:
                list_sentences.append(Sentence(line))
        return Document(list_sentences, file_id)

    def gen_sentences_str(self):
        return [sentence.string for sentence in self.list_sentences]

class Sentence(object):
    def __init__(self, string, vector = None):
        self.string = string
        self.vector = vector
        self.length = string.count(" ")
        self.sentece_id = -1

import numpy as np
import time
import pickle

def load_duc04():
    duc_ref_path = "/Users/HyNguyen/Documents/Research/Data/duc2004/duc2004_results/ROUGE/task2_model/models/2"
    duc_doc_path = "/Users/HyNguyen/Documents/Research/Data/duc2004/DUC2004_Summarization_Documents/duc2004_testdata/tasks1and2/duc2004_tasks1and2_docs/docs"
    list_ref_file = os.listdir(duc_ref_path)

    clusters = []

    for cluster_id in os.listdir(duc_doc_path):
        if cluster_id[0] == ".":
            continue
        cluster = Cluster.load_from_folder_duc04(cluster_id,duc_doc_path+ "/"+cluster_id,duc_ref_path)
        cluster.gen_data_for_rouge_method("data/model/duc04", "data/peer/duc04")
        clusters.append(cluster)
    print(len(clusters))

def load_duc05():
    duc_ref_path = "/Users/HyNguyen/Documents/Research/Data/duc2005/results/ROUGE/models"
    duc_doc_path = "/Users/HyNguyen/Documents/Research/Data/duc2005/DUC2005_Summarization_Documents/duc2005_docs"

    list_ref_file = os.listdir(duc_ref_path)

    clusters = []

    for i,cluster_id in enumerate(os.listdir(duc_doc_path)):
        if cluster_id[0] == ".":
            continue
        cluster = Cluster.load_from_folder_duc05(cluster_id,duc_doc_path+ "/"+cluster_id,duc_ref_path)
        cluster.gen_data_for_rouge_method("data/model/duc05", "data/peer/duc05")
        clusters.append(cluster)

    print(len(clusters))



if __name__ == "__main__":

    load_duc04()



