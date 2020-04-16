#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:18:46 2018

@author: sema
"""

import os
import numpy as np
import pandas as pd
from preprocessing.data_processing_object_prog import *
from preprocessing.process_data_for_nlp import *

from gensim.models import Word2Vec
from gensim.similarities import MatrixSimilarity
from sklearn.metrics import pairwise_distances 
from scipy.spatial.distance import cosine

# Set the correct path
os.chdir("/home/sema/thesis/basicCode/data_processed")
conso = pd.read_pickle('conso_ad.p')
context = pd.read_pickle("meal_context_clean.p")
conso = conso.reset_index()


#process data
I,U = get_consumption_sequence(conso, context)
meals = get_all_meals(I, 'codsougr')
all_meals = [m[1] for m in meals]
corpus, dictionary = get_corpus(all_meals)

train_corpus, test_corpus = split_corpus(corpus, 0.2)

word2vec = Word2Vec(size=10, min_count = 50, sg=0)
word2vec.build_vocab(all_meals[:30000])
word2vec.train(all_meals, total_examples = word2vec.corpus_count, epochs=100)

m = word2vec.wv.syn0
index = word2vec.wv.index2word
dist_out = 1 -pairwise_distances(m, metric = 'cosine')
dist = pd.DataFrame(dist_out, index = index, columns = index)

os.chdir("/home/sema/thesis/basicCode/results")
dist.to_pickle('dist_matrix_substitutability_word2vec.p')