import numpy as np
from utils.ultis import n_grams, cosine_sim
import pickle
class TfidfVectorizer():
    @classmethod
    def tf_Computer(cls, index):
        term_doc_mx = index.get_postingList()
        tf_Vector = term_doc_mx.T
        return np.log10(tf_Vector + 1)

    @classmethod
    def idf_Computer(cls, index, smooth_idf=True):
        term_doc_mx = index.get_postingList()
        N = len(index.tokenized_docs)

        df_Vector = (term_doc_mx > 0).sum(1)
        if smooth_idf:
            return np.log10((1 + N)/(1 + df_Vector)) + 1
        else:
            return np.log10(N / df_Vector)

    @classmethod
    def _build(cls):
        cls.tf_Computer()
        cls.idf_Computer()

    def __init__(self, dataset, index):
        self.dataset = dataset
        self.index = index

        self.tf_ = TfidfVectorizer.tf_Computer(self.index)
        self.idf_ = TfidfVectorizer.idf_Computer(self.index, smooth_idf=True)

        self.tfidf_ = self.tf_ * self.idf_


    def query(self, q, topk=10):
        q_tokenized = n_grams(q, len(list(self.index.vocab.keys())[0].split()))

        vocab = self.index.vocab
        inverted_index = self.index.inverted_index

        q_tf = np.zeros((len(vocab), ))
        for token in q_tokenized:
            if token in vocab:
                token_id = vocab[token]
                q_tf[token_id] += 1

        q_tf = np.log10(q_tf + 1)
        q_tfidf = q_tf * self.idf_

        return cosine_sim(q_tfidf, self.tfidf_, topk)
        # return q_tf, q_tfidf, cosine_sim(q_tfidf, self.tfidf_, topk)