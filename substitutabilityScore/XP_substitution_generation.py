#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sema

Generate substitutability scores from a .p file

1. Preprocess data 
2. Compute substitutabilityScore 
    score =  [jaccardIndex, jaccardIndex2]
3. Output in the form of a dict

"""
import os
import pickle
import datetime
import logging

date = datetime.datetime.now().strftime('%y_%m_%d_%H_%M')
filename = 'substitutabilityScore/logs/XP_' + date + '.log'
os.makedirs(os.path.dirname(filename), exist_ok=True)
logging.getLogger().setLevel(logging.DEBUG)
logging.basicConfig(filename=filename, 
                    format = '%(asctime)s %(message)s', 
                    level=logging.INFO)

from substitutability import jaccardIndex, openPickleFile, saveToDict, main
print(os.getcwd())

filepath_dico = 'data/dict_cod.p'
filepath_data = 'data/conso_ad.p'
level = 'codsougr'
max_meal = 10
score = jaccardIndex

### Import data
dict_cod = openPickleFile(filepath_dico)
conso = openPickleFile(filepath_data)
dico = dict_cod[level]


logging.info('Data imported...')

logging.info('Computing subScores...')
res = main(conso, level, dico, max_meal, score)
res_dict = saveToDict(res)

logging.info('Saving the results...')
with open('substitutabilityScore/results/subScoreAllMeals1.p', 'wb') as handle:
    pickle.dump(res_dict, handle)

logging.info('Generating result table...')