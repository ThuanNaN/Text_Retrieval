from utils.ultis import n_grams
import numpy as np
from prettytable import PrettyTable

class InvertedIndex:
    """
    Inverted Index class.
    """

    def __init__(self, data, n_grams_token=1):
        self.data = data

        self.tokenized_docs = [n_grams(sentence['content'], n_grams_token) for sentence in self.data]

        corpus = []
        for doc in self.tokenized_docs:
            corpus.extend(doc)
        corpus = sorted(list(set(corpus)))

        self.vocab = {w: i for i, w in enumerate(corpus)}

        self.inverted_index = dict()
        for term in self.vocab:
            obj = {}
            obj['doc_freq'] = 0
            obj['posting_list'] = np.zeros(len(self.data), dtype=np.int64)
            self.inverted_index[term] = obj


    def indexer(self):
        """
        Create inverted index for `Dataset`
        """
        for i in range(len(self.tokenized_docs)):
            for token in self.tokenized_docs[i]:
                term = self.inverted_index[token]
                term['posting_list'][i] += 1
                term['doc_freq'] = (term['posting_list'] != 0).sum()

        return self.inverted_index


    def get_df(self):
        """
            Return array that has shape (n_terms, )
        """
        return np.array([self.inverted_index[term]['doc_freq'] for term in self.inverted_index])


    def get_postingList(self):
        """
            Return array that has shape (n_terms, n_docs)
        """
        return np.array([self.inverted_index[term]['posting_list'] for term in self.inverted_index])

    def __repr__(self):
        tab = PrettyTable(['Token', 'DocFreq', 'Posting List'])
        for token in self.inverted_index:
            tab.add_row([token, self.inverted_index[token]['doc_freq'], self.inverted_index[token]['posting_list']])
        print(tab)
        return f"Total {len(self.inverted_index)} token(s)"      

