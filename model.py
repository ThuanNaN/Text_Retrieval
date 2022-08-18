
from dataset import Dataset
from invertedIndex import InvertedIndex

from utils.data_preprocessing import preprocessing, preprocess_res
from utils.ultis import n_grams, AP, cosine_sim
import numpy as np
from numpy.linalg import svd, norm
from numpy import dot
import numpy as np
import json
import pickle

from tfidf import TfidfVectorizer

def lsi(path_docs, path_query, topk):

    data = Dataset(path_docs)

    index = InvertedIndex(data, 1)
    
    index.indexer()
    termdoc = index.get_postingList()
    vocab = index.vocab

    
    with open(path_query) as f:
        queries = f.readlines()

    query_preprocessed = []
    for q in queries:
        query_id, content = q.split("\t")
        preprocessed_query = preprocessing(content)
        query_preprocessed.append(preprocessed_query)


    tokenized_queries = [n_grams(sen, 1) for sen in query_preprocessed]

    termqr = np.zeros((len(tokenized_queries), len(vocab)))

    for i in range(len(tokenized_queries)):
        for w in tokenized_queries[i]:
            if w in vocab:
                token_id = vocab[w]
                termqr[i][token_id] +=1
    termqr = np.array(termqr)

    S, Z, Ut = svd(termdoc, full_matrices=False)
    Z = np.diag(Z)
    vector_doc = dot(Z, Ut)
    vector_term = dot(S, Z)

    vector_doc = vector_doc.transpose()
    # with open('./vector_doc.pickle', 'wb') as f:
    #     pickle.dump(vector_doc, f)

    queries = []
    for i in range(len(termqr)):
        q = np.zeros((1400, ))

        term_freq = np.where(termqr[i] > 0)[0]

        for j in term_freq:
            q += vector_term[j] * termqr[i][j]  
        
        if len(term_freq) > 0:
            q = q / len(term_freq)
            
        queries.append(q)
    queries = np.array(queries)

    # with open('./vector_queries.pickle', 'wb') as f:
    #     pickle.dump(queries, f)


    rs = []
    for query in queries:
        doc_id, cosine = cosine_sim(query, vector_doc, topk=topk)
        rs.append(doc_id)
    rs = np.array(rs)
    rs+=1
    return rs

def vsm(path_docs, path_query, topk):

    data = Dataset(path_docs)
    index = InvertedIndex(data, 1)
    index.indexer()
    tfidf_vectorizer = TfidfVectorizer(data, index)

    with open(path_query) as f:
        queries = f.readlines()

    query_preprocessed = []
    for q in queries:
        query_id, content = q.split("\t")
        preprocessed_query = preprocessing(content)
        query_preprocessed.append(preprocessed_query)

    rs = []
    for q in query_preprocessed:
        idx, cosine = tfidf_vectorizer.query(q, topk)
        rs.append(idx)

    rs = np.array(rs)
    rs +=1

    return rs
    # res = preprocess_res("./TEST/RES")
    # ap = AP(rs, res).mean()
    # print(ap)



# path_docs = "./Cranfield"
# path_query = "./TEST/query.txt"
# vsm(path_docs,path_query )