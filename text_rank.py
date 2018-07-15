# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:55:36 2018

@author: Ethan Beaman
"""

import numpy as np
from operator import itemgetter
from collections import Counter
from gensim.utils import simple_tokenize
from sklearn.decomposition import TruncatedSVD
from pymagnitude import Magnitude
from nltk import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

class Summarizer:
    def __init__(self, vec_file):
        self.word_vec = Magnitude(vec_file)
    
    def _flatten(self, l):
        return [item for sublist in l for item in sublist]
    
    def preprocess(self, text):
        seen = set()
        sents = []
        for sent in text:
            processed = tuple(simple_tokenize(sent))
            if processed not in seen:
                sents.append(processed)
                seen.add(processed)
        return sents
    
    def sent2vec(self, sents, word_vec, a=0.001):
        words = self._flatten(sents)
        w = Counter(words)
        dim = word_vec.dim
        pw = {word: count/len(w) for word, count in w.items()}
        X = np.zeros((len(sents), dim))
        for i, sent in enumerate(sents):
            s = len(sent)
            vs = np.zeros(dim)
            for word in sent:
                vs += (a/(a + pw[word])) * word_vec.query(word)
            vs /= s
            X[i] = vs
        svd = TruncatedSVD(n_components=1, n_iter=10, random_state=0)
        svd.fit(X)
        pc = svd.components_
        X -= X.dot(pc.T).dot(pc)
        return X.clip(min=0)  
    
    def _build_similarity_matrix(self, X):
        # Create an empty similarity matrix
        return cosine_similarity(X)
    
    def _pagerank(self, A, eps=0.0000001, d=0.85):
        rsums = A.sum(axis=1).reshape(-1, 1)
        A /= rsums
        P = np.ones(len(A)) / len(A)
        i = 0
        while True:
            new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
            delta = abs(new_P - P).sum()
            if delta <= eps:
                print(i)
                return new_P
            P = new_P
            i += 1
    
    def textrank(self, sentences, top_n=5):
        """
        sentences = a list of sentences [(w11, w12, ...), (w21, w22, ...), ...]
        top_n = how may sentences the summary should contain
        """
        S = self._build_similarity_matrix(sent_vec) 
        sentence_ranks = self._pagerank(S)
        # Sort the sentence ranks
        sorted_ranks = sorted(enumerate(sentence_ranks), key=lambda item: -item[1])
        ranked_sentence_indexes = [idx for idx, rank in sorted_ranks]
        selected_sentences = sorted(ranked_sentence_indexes[:top_n])
        summary = itemgetter(*selected_sentences)(sentences)
        return summary
    

if __name__ == 'main':
    with open(r'C:\Users\Ethan Beaman\Documents\test.txt', 
              encoding='utf-8') as f:
        tests = f.read().splitlines()
        summarizer = Summarizer('vecfile.magnitude')
        for test in tests:
            data = summarizer.preprocess(sent_tokenize(test))
            sents = [s for s in data if len(s) > 5]
            sent_vec = summarizer.sent2vec(sents, summarizer.word_vec)
            for idx, sentence in enumerate(summarizer.textrank(sents, 7)):
                print("%s. %s" % ((idx + 1), ' '.join(sentence)))

