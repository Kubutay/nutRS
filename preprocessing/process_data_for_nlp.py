#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 11:39:38 2018

@author: Sema

Prepare data for doc2vec users 
"""

from data_processing_object_prog import MealSequence, Food, Meal
import os
import pickle

from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
from random import shuffle, seed

from gensim.models.doc2vec import TaggedDocument

    
#### FUNCTIONS
def flatten_list(nested_list):
    return [x for liste in nested_list for x in liste]

def doc_to_token(docs):
    """
    docs (list of nested lists of words)
    """
    dictionary = corpora.Dictionary(docs)
    corpus = [(i, dictionary.doc2bow(text)) for i, text in enumerate(docs)]
    return corpus, dictionary

def process_corpus(corpus, dictionary):
    processed_corpus = []
    for nomen, text in corpus:
        doc = TaggedDocument([dictionary[word_id] for word_id, occ in text], [str(nomen)]) 
        processed_corpus.append(doc) 
    return processed_corpus 

def remove_duplicates(corpus):
    return list(set([tuple(doc) for doc in corpus]))  

def split_corpus(corpus, ratio):
    n = int(ratio * len(corpus))
    train = corpus[:n]
    test = corpus[n:]
    return train, test  

def get_corpus(docs):
    corpus, dictionary = doc_to_token(docs)
    processed_corpus = process_corpus(corpus, dictionary)
    return processed_corpus, dictionary


def get_meal_by_cod(conso_seq, cod_name): 
    """
    Extract the meals of the database 
    
    input : dict of meal_seq object
    output : nested list of strings
    """
    meals = [(getattr(m, cod_name), m.tyrep) for meals in conso_seq.values() for m in meals.meal_list]
    meals = [x for x in meals if x[0]]
    return meals

def get_corpus_meal(docs):
    """
    docs 
    """
    docs = [x[0] for x in docs]
    return get_corpus(docs)
    

def main():
    pass


if __name__ == '__main__':
    main() 
    
    

#def main():
#    ###### IMPORT DATA
#    path = os.path.join(os.getcwd(), 'data')
#    os.chdir(path) 
#    
#    with open('conso_seq.p', 'rb') as handle:
#        U = pickle.load(handle)
#        
#        
#    #Get user_doc   
#    docs_codsougr = [[nomen, flatten_list(meals.codsougr_name_list)] for nomen, meals in U.items()]
#    docs_codal = [[nomen, flatten_list(meals.codal_name_list)] for nomen, meals in U.items()]
#    docs_codgr = [[nomen, flatten_list(meals.codgr_name_list)] for nomen, meals in U.items()]
#    
#    A_codsougr  = get_corpus(docs_codsougr)
#    A_codal = get_corpus(docs_codal)
#    A_codgr = get_corpus(docs_codgr)
#    
#    #Get meal doc
#    meal_codsougr = get_meal_by_cod(U, 'meal_codsougr_name')
#    meal_codgr = get_meal_by_cod(U, 'meal_codgr_name') 
#    meal_codal = get_meal_by_cod(U, 'meal_codal_name')
#    
#    
#    with open('users_doc.p', 'wb') as f:
#        pickle.dump([A_codgr, A_codsougr, A_codal], f)
#    
#    with open('meals_doc.p', 'wb') as f:
#        pickle.dump([meal_codgr, meal_codsougr, meal_codal], f)