#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sema

See paper "Investigating substitutability..." for the pseudocode. 

"""
import logging
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

### Functions
#use frequent itemsets for associations generation

# Preprocessing 
def openPickleFile(filename):
    with open(filename, 'rb') as handle:
        res = pickle.load(handle)
    return res

def getMeals(df, level, tyrep=[0]):
    """
    From a consumption table, get the meals. 
    
    Input : 
        df (pd.DataFrame) each row is an entry of a food diary
        tyrep (list of int) if 0, all meals are considered
        cod (str) must take value in []
    
    Output : 
        meals (list of lists)
    """
    df_ = pd.DataFrame(df.groupby(['nomen','jour','tyrep'])[level].unique())
    
    if 0 not in tyrep:
        df_ = pd.DataFrame(df_[df_.index.get_level_values('tyrep').isin(tyrep)])
    seq = lambda x: [list(aliment) for repas in x for aliment in repas] 
    meals = seq(df_.values)
    return meals
    

# SubstitutabilityScore funct ions
def isLinked(tuple1, tuple2):
    """
    These two nodes are linked if only one item is changed. 
    
    Return the edge (tuple1, tuple2)
    """
    set1 = set(tuple1)
    set2 = set(tuple2)
    len1 = len(tuple1)
    len2 = len(tuple2)
    
    if (len1 == len2) & (len(set.intersection(set1, set2)) == len1 - 1):
        return (tuple1, tuple2)
    
    elif (min(len1,len2) == max(len1, len2) - 1) & (len(set.intersection(set1, set2)) ==  min(len1,len2)):
        return (tuple1, tuple2)

def getEdges(nodelist):
    edges = []
    for i in range(len(nodelist)):
        for j in range(i, len(nodelist)):
            edge = isLinked(nodelist[i], nodelist[j])
            if edge is not None:
                edges.append(edge)
    return edges

def getGraph(nodelist):
    
    G = nx.Graph()
    logging.info('Computing edges...')
    edges = getEdges(nodelist)
    
    G.add_nodes_from(nodelist)
    logging.info('Successfully created the nodes.')
    G.add_edges_from(edges)
    logging.info('Successfully created the edges.')
    
    return G 

def getCliques(G, min_node_len):
    return [x for x in list(nx.find_cliques(G))  if len(x) > min_node_len] 
    

def getContext(clique):
    """
    Get the intersection of all elements in a clique
    clique (list)
    
    return a tuple
    """
    return tuple(set.intersection(*[set(x) for x in clique])) 

def isSubClique(clique):
    """
    Checks if a clique is a substitutable clique. 
    clique : list
    return a boolean
    """
    context = getContext(clique)
    max_meal = max(len(x) for x in clique)
    if max_meal - len(context) == 1:
        return True
    else:
        return False
    
def getSubClique(cliques):
    """
    Get all substitutable cliques from a list of cliques.
    """
    return [clique for clique in cliques if isSubClique(clique)] 

def getSubset(clique, context):
    """
    Get the substitutable set of a clique i.e the complement of the intersection all sets.
    """
    subset = [list(set(x).difference(set(context))) for x in clique]
    flat_subset = [x for sublist in subset for x in sublist]
    
    if len(subset) == len(flat_subset):
        return flat_subset
    elif len(flat_subset) == len(subset) -1:
        flat_subset.append('')
        return flat_subset
    
def getAllSubsets(subcliques):
    """
    Get all the substitutable sets from the substitutable cliques. 
    """
    S = {}
    for subclique in subcliques:
        context = getContext(subclique)
        subset = getSubset(subclique, context)
        S[context] = subset
    return S 
    

def getItemContextSet(S, itemlist):
    """
    Get the list of contexts of which the itemlist appears in the substitutable set. 
    
    S (dict) : the key is the context and the value is the substitutable set.
    Return the list 
    """
    return [c for c, s in S.items() if set(itemlist).issubset(set(s))] 

def jaccardIndex(S, item1, item2):
    contextSet1 = getItemContextSet(S, [item1]) 
    contextSet2 = getItemContextSet(S, [item2])
    i = len(set.intersection(set(contextSet1), set(contextSet2))) 
    u = len(set.union(set(contextSet1), set(contextSet2))) 
    ci = len([x for x in contextSet2 if item1 in x])
    cj = len([x for x in contextSet1 if item2 in x])
    if u <= 0:
        return 0
    else:
        return i / (u + ci + cj)

def getWeightedContextSet(contextSet, context_weights):
    """
    Get the weighted count of a context set.
    """
    return sum([context_weights[c] for c in contextSet])

def jaccardIndex2(S, item1, item2, context_counts):
    contextSet1 = getItemContextSet(S, [item1]) 
    contextSet2 = getItemContextSet(S, [item2])
    interContextSet = set.intersection(set(contextSet1), set(contextSet2))
    unionContextSet = set.union(set(contextSet1), set(contextSet2))
    ci = [x for x in contextSet2 if item1 in x]
    cj = [x for x in contextSet1 if item2 in x]    
    
    i = getWeightedContextSet(interContextSet, context_counts) 
    u = getWeightedContextSet(unionContextSet, context_counts) 
    wi = getWeightedContextSet(ci, context_counts)
    wj = getWeightedContextSet(cj, context_counts)

    #print(i, u, wi, wj)
    if u <= 0:
        return 0
    else:
        return i / (u + wi + wj)
    
def getContextWeights(L, cliques): 
    m_freq = Counter([meal for meal in L])
    return {getContext(clique):sum([m_freq[meal] for meal in clique]) for clique in cliques}
    
    
def substitutabilityMatrix(S, dico, score):
    keys = sorted(dico.keys(), key = int)
    keysInt = [int(i) for i in keys]
    subs = pd.DataFrame(0, 
                        index=keysInt, 
                        columns=keysInt) 
    for i in list(subs.index):
        for j in list(subs.index):
            subs.loc[i,j] = score(S,i,j) 
    return subs 

def computeSubstitutability(meals, dico, max_meal, score):
    """
    Compute subsitutability score from a list of meals (tuple). 
    
    Return a table of substitutability (pd.dataframe)
    
    """
    # What was the use of the line below ???
    #meals = [x for x in Counter(L)] 
    logging.info('Number of meals: {}'.format(len(meals)))
    meals_ = [tuple(sorted(meal,key=int)) for meal in meals]
    meals_ = list(set(meals_))
    
    nodes = [x for x in meals_ if len(x) <= max_meal] 
    logging.info('Number of nodes: {}'.format(len(nodes)))
    
    logging.info("Getting the graph...")
    G = getGraph(nodes)
    
    logging.info("Computing the cliques...")
    cliques = getCliques(G, 1)
    logging.info('Getting the substitutable cliques...')
    subCliques = getSubClique(cliques) 
    S = getAllSubsets(subCliques) 
    logging.info('Computing substitutability...')
    #context_weights = getContextWeights(L, cliques)
    s = substitutabilityMatrix(S, dico, score)
    return s


# Presentation of results
def saveToDict(df):
    """
    Convert the result dataframe of substitutability scores to a dict 
    where key = (a,b) and value = subScore
    
    a,b (int)
    """    
    res_dict = df.to_dict()
    R = {(idx,col):val for idx, series in res_dict.items() for col, val in series.items() if idx != col}
    return R

def results_to_list(res_pd, dico, n):
    RES = pd.DataFrame()
    for x in list(res_pd.index):
        X = [dico[y] + ' ' + '{:.4f}'.format(res_pd.loc[x,str(y)]) if res_pd.loc[x,str(y)] < 1 else dico[y] for y in list(res_pd[x].nlargest(n).index)]
        #print([dict_codsougr[y] for y in list(res_pd[x].nlargest(5).index)]) 
        RES[X[0]] = X[1:]
    return RES.T

def subtitutabilityResultsLatex(df1, df2, k, name1, name2):
    """Generate latex code for printing rankings for the two matrices. 
    
    Input :
        k : top-k list ranking. 
    """
    items = df1.index.tolist()
    ranks = np.arange(1,k+1)
    idx = pd.MultiIndex.from_product([items, ranks], names = ['Item', 'rank'])
    col = pd.MultiIndex.from_product([[name1, name2], ['substitute', 'score']])
    res = pd.DataFrame(index=idx, columns=col)
    
    for item in items:
        print(item)
        l1 = df1.loc[item].sort_values(ascending=False).nlargest(k)
        l2 = df2.loc[item].sort_values(ascending=False).nlargest(k)
        
        res.loc[(item), (name1, 'substitute')] = l1.index.values
        res.loc[(item), (name1, 'score')] = l1.values
        
        res.loc[(item), (name2, 'substitute')] = l2.index.values
        res.loc[(item), (name2, 'score')] = l2.values
    return res

def main(df, level, dico, max_meal, score, tyrep=[0]):
    logging.info('Preprocessing food consumption data to meals...')
    meals = getMeals(df, level, tyrep)
    logging.info('Number of meals: {}'.format(len(meals)))
    logging.info('----------------------------------------')
    logging.info('Computing the substitutability scores...')
    res = computeSubstitutability(meals, dico, max_meal, score)
    logging.info('FINISHED')
    return res

#if __name__ == '__main__':
#    main()