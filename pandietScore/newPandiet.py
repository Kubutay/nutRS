#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:52:16 2019

@author: sema
Compute pandiet score given a dataframe of consumption of a single user.
Nomen, jour, tyrep, cod, qte_nette


Parameters : 
    composition table (pd.DataFrame) index cod, values are numeric.
    references table (pd.DataFrame) index nutrientName, values are numeric
    
"""
import pandas as pd


def addNutrientValues(df, composition, df_socio, level):
    """
    Compute the nurtitional intake of each food item with the nutritional 
    composition table and the quantity. 
    
    Input:
        df(pd.DataFrame) multindex, consumption df
        composition (df.DataFrame)
    
    Output:
        df (df.DataFrame) with nutrient columns in addition
        
    """
    cols = composition.columns.tolist()
    if isinstance(df.index, pd.core.index.MultiIndex):
        df.reset_index(inplace=True)
    df1 = pd.merge(df, composition, on=level)
    #df1.update(df1[cols].mul(df1.qte_nette,0))
    for col in cols:
        df1[col] *= df1['qte_nette'] 
    df1[cols] *= 0.01
    
    ## Add Nutrient Value By User Weight for proteines and mg
    cols_ = ['mg','mg_dis','proteines_d','proteines_N_d']
    cols_w = [col+'_kg' for col in cols_]
    df1 = pd.merge(df1, df_socio[['nomen', 'poidsm']], on='nomen')
    df1[cols_w] = df1[cols_].copy()
    for col in cols_w:
        df1[col] /= df1['poidsm'] 
    cols += cols_w
    return df1, cols
