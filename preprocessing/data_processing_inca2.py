#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:27:26 2018

@author: Sema

Process food diary data for NLP models
"""

import os
import pickle
import pandas as pd

path = os.getcwd()

# Import data 
dataPath = path + '/rawData'
os.chdir(dataPath) 

users = pd.read_csv("Table_indiv.csv", sep=";", 
                       header = 0, 
                       index_col = "nomen", 
                       dtype='str', encoding='ISO-8859-1').sort_index()

consumption = pd.read_csv("Table_conso.csv", sep=";", 
                           header = 0, 
                           index_col = ["nomen", "nojour", "tyrep"],
                           dtype='str', 
                           na_values= ['x']).sort_index()

nomenclature = pd.read_csv("Nomenclature_3_en1.csv", 
                           sep=";", header = 0, 
                           index_col = "codal", 
                           dtype='str', encoding='ISO-8859-1').sort_index()

meal_context = pd.read_csv("Table_repas.csv", sep=";", 
                    header = 0, 
                    index_col = [0, "nojour", "tyrep"], 
                    dtype = 'str', encoding='ISO-8859-1').sort_index()

processedPath = path + '/processedData'
os.chdir(processedPath)

########## DATA CLEANING
## Users table
users.drop('sexeps', axis = 1, inplace=True) # drop useless var 
users.dropna(axis = 1, how = 'all', inplace = True) # drop empty cols 

adults = users[users['ech']== '1']
kids = users[users['ech']== '2']

adults.to_pickle('adults.p')
kids.to_pickle('kids.p')

## Nomenclature table
nom = nomenclature[nomenclature.codgr != '45'] 
nom_codgr = nom.groupby(by='codgr')['libgren'].unique().to_dict()
nom_sougr = nom.groupby(by='codsougr')['libsougren'].unique().to_dict()
nom_codal = nom.groupby(level='codal')['libal'].unique().to_dict()

dict_codgr = {key:nom_codgr[key][0] for key in nom_codgr.keys()}  
dict_sougr = {key:nom_sougr[key][0] for key in nom_sougr.keys()} 
dict_codal = {str(key):nom_codal[key][0] for key in nom_codal.keys()} 

dict_cod = {'codgr': dict_codgr, 'codsougr':dict_sougr, 'codal':dict_codal}
with open('dict_cod.p', 'wb') as handle:
    pickle.dump(dict_cod, handle, protocol=pickle.HIGHEST_PROTOCOL)


## Meal_context table
# Match days of consumption to days of the week
rep_jour = pd.DataFrame(meal_context.groupby(level=['nomen', 
                                                    'nojour', 
                                                    'tyrep']).first()['nomjour'])
dict_jour = {'lundi':1, 'mardi':2, 'mercredi':3, 'jeudi':4, 'vendredi':5, 
             'samedi':6, 'dimanche':7}
rep_jour['jour'] = rep_jour.nomjour.replace(dict_jour)
meal_context['jour'] = rep_jour.jour.copy() 
meal_context.reset_index(inplace=True)
meal_context.set_index(['nomen', 'jour', 'tyrep'], inplace=True) 
meal_context.sort_index(level = ['nomen', 'jour'], inplace=True)
rep = meal_context[meal_context.temrep == '0']
rep.to_pickle('meal_context_clean.p') 

## Consumption table
consumption['codsougr'] = consumption.codgr + consumption.sougr 
consumption['codal_name'] = consumption['codal'].map(dict_codal)
consumption['codgr_name'] = consumption['codgr'].map(dict_codgr)
consumption['codsougr_name'] = consumption['codsougr'].map(dict_sougr)

conso_unsort = consumption[(consumption.codgr != '45') & 
                           (consumption.codgr != '44')]
conso_unsort['jour'] = rep_jour.jour.copy()
conso_unsort.reset_index(inplace=True) 
conso_unsort.set_index(['nomen', 'jour', 'tyrep'], inplace=True)
conso = conso_unsort.sort_index(level = ['nomen','jour'])

conso_ad = conso.loc[conso.index.get_level_values('nomen').isin(adults.index)].copy()
conso_kids = conso.loc[conso.index.get_level_values('nomen').isin(kids.index)].copy()

conso_ad.to_pickle('conso_ad.p') 
conso_kids.to_pickle('conso_kids.p') 
conso_ad.to_csv('conso_ad.csv') 