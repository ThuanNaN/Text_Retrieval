from nltk import ngrams
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

def n_grams(text, n):
    if n == 1:
        return text.split()
    n_grams_text = ngrams(text.split(), n)
    return [" ".join([str(token) for token in list(l_token)]) for l_token in n_grams_text]


def AP(q_results, res):
    AP = []

    for i in range(len(q_results)):
        pos = 0
        P = 0
        for j in range(len(q_results[i])):
            if (q_results[i][j]) in res[i]['rel_ans']:
                pos += 1
                P += pos / (j + 1)

        q_AP = P / pos
        AP.append(q_AP)

    return np.array(AP)

def cosine_sim(q_vec, c_vec, topk=10):
    cos_sim = []
    for i in range(len(c_vec)):
        cos_sim.append(cosine_similarity(c_vec[i].reshape(1, -1), q_vec.reshape(1, -1)))

    cos_sim = np.array(cos_sim).reshape(-1)
    topk_ans = cos_sim.argsort()[-topk:][::-1]
    return topk_ans, cos_sim

def save_result(path_folder, data):
    
    isExist = os.path.exists(path_folder)
    if not isExist:
        os.makedirs(path_folder)

    for i in range(len(data)):
        path_save = os.path.join(os.getcwd(),path_folder.split("/")[-1],str(i+1)+".txt")
        
        with open(path_save,"w") as f:
            for j in data[i]:
                s = str(i+1)+" "+ str(j) + "\n"
                f.write(s)
