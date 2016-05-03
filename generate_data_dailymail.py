__author__ = 'HyNguyen'

import xml.etree.ElementTree as ET
import unicodedata

if __name__ == "__main__":

    model_path = "data/model/dailymail"
    peer_path = "data/peer/dailymail"

    file_path = "data/data_dailymail.xml"
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []
    print root
    for child in root._children:
        sentences_doc = child._children[0].text
        sentences_ref = child._children[1].text
        if len(sentences_doc) < len(sentences_ref):
            print("co gi do sai sai")
            print sentences_doc
            print sentences_ref
        else:
            data.append((sentences_doc.count(" "), sentences_ref.count(" "), sentences_doc,sentences_ref))
            if isinstance(sentences_doc, unicode):
                sentences_doc = unicodedata.normalize('NFKD', sentences_doc).encode('ascii','ignore')
            if isinstance(sentences_ref, unicode):
                sentences_ref = unicodedata.normalize('NFKD', sentences_ref).encode('ascii','ignore')
            sentences_doc = sentences_doc.split("\n")
            sentences_ref = sentences_ref.split("\n")
            with open(model_path + "/DAILY.MAIL."+ child.attrib["id"], mode="w") as f:
                for sent in sentences_ref:
                    if len(sent) > 5:
                        f.write(sent + "\n")
            for i,sent in enumerate(sentences_doc):
                if len(sent) > 5:
                    with open(peer_path + "/DAILY.MAIL."+ child.attrib["id"] + "." + str(i), mode="w") as f:
                        f.write(sent)
    print(len(data))