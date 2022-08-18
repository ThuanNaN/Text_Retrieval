
import re
import nltk
from nltk.stem import PorterStemmer
import spacy
import json
import os

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
nlp.max_length=5000000

stemmer = PorterStemmer()


with open("./utils/contractions_dict.json") as f:
    contractions_dict = json.load(f)
    

contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))


def expand_contractions(text,contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

  
def clean_text(text):
    text=re.sub('\w*\d\w*','', text)
    text=re.sub('\n',' ',text)
    text=re.sub('\t',' ',text)
    text=re.sub(r"http\S+", "", text)
    text=re.sub('[^a-z]',' ',text)
    text = re.sub(' +',' ', text)
    return text


def rm_stop_words(text, stemming=False):
    if stemming:
        return ' '.join([stemmer.stem(str(token)) for token in list(nlp(text)) if (token.is_stop==False)])
    return ' '.join([token.lemma_ for token in list(nlp(text)) if (token.is_stop==False)])

def rm_single_character(text):
    return ' '.join([token for token in text.split() if (len(token) > 1)])


def preprocessing(text):
    c = text.lower()
    c = expand_contractions(c)
    c = clean_text(c)
    c = rm_stop_words(c, stemming=True)
    c = rm_single_character(c)
    return c


def preprocess_res(res_path):
    files_path = os.listdir(res_path)
    files_path = sorted(files_path, key=lambda path:(int(path.split(".")[0])))
    res = []
   
    for i,file in enumerate(files_path):
        path = os.path.join(os.getcwd(),"TEST","RES",file)
        obj = {}
        with open(path, "r") as f:
            rels = [rel.strip().split("\t") for rel in f.readlines()]
            lst = []
            for j in rels:
                if len(j) == 1:
                    pass
                else: 
                    if j[1] != '-1':
                        lst.append(j[0].split(" ")[1])

            obj['query_id'] = str(file.split(".")[0])
            obj['rel_ans'] = lst
            res.append(obj)

    return res
