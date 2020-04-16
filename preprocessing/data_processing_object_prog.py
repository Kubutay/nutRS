#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:42:37 2018

@author: Sema
"""
import pandas as pd
from operator import attrgetter

class User(object):
    def __init__(self, nomen, sexe_ps, v2_age, poidsd, poidsm, bmi, 
                 menopause, regmaig, regmedic, regrelig, regvegr, regvegt, 
                 ordi, bonalim, agglo9):
        self.nomen = int(nomen)
        self.sexe_ps = int(sexe_ps)
        self.v2_age = int(v2_age)
        self.poidsd = float(poidsd)
        self.poidsm = float(poidsm)
        self.bmi = float(bmi)
        self.menopause = int(menopause)
        
        self.regmaig = int(regmaig)
        self.regmedic = int(regmedic)
        self.regrelig = int(regrelig)
        self.regvegr = int(regvegr)
        self.regvegt = int(regvegt)
        
        self.ordi = float(ordi)
        self.bonalim = int(bonalim)
        self.agglo9 = int(agglo9)
    
    def as_dict(self):
        return {'nomen':int(self.nomen), 
                'sexe_ps':int(self.sexe_ps), 
                'v2_age':int(self.v2_age),
                'poidsd':self.poidsd, 
                'poidsm':self.poidsm, 
                'bmi':self.bmi,
                'menopause':int(self.menopause), 
                'regmaig':int(self.regmaig), 
                'regvegr':int(self.regvegr)}
        
        
class Food(object):
    def __init__(self, codgr, codsougr, codal, codgr_name, codsougr_name, 
                 codal_name, qte_brute, qte_nette, typal2, typal3):
        self.codgr = int(codgr)
        self.codsougr = int(codsougr)
        self.codal = int(codal)
        self.codgr_name = str(codgr_name)
        self.codsougr_name = str(codsougr_name)
        self.codal_name = str(codal_name)
        self.qte_brute = float(qte_brute)
        self.qte_nette = float(qte_nette)
        self.typal2 = typal2
        self.typal3 = typal3
               
class Meal(object):
    """
    A meal is characterized by the set of food items, the type of meal, the day
    the place, the companion, the duration. 
    
    food_set (list) : list of Food objects
    """
    def __init__(self, food_set, tyrep, day, 
                 place = None, companion = None, duration = None):
        self.meal = food_set
        self.tyrep = tyrep
        self.day = day
        self.place = place 
        self.companion = companion
        self.duration = duration
        self.codgr = [x.codgr for x in self.meal]
        self.codsougr = [x.codsougr for x in self.meal]
        self.codal = [x.codal for x in self.meal]
        self.codgr_name = [x.codgr_name for x in self.meal]
        self.codsougr_name = [x.codsougr_name for x in self.meal]
        self.codal_name = [x.codal_name for x in self.meal]
    
    def findItemIndex(self, level, value):
        return next(i for i,m in enumerate(self.meal) 
                    if getattr(m, level) == value)
        
#    def addItem(self):
#        
#        return self
#    
#    def deleteItem(self):
#        return self

class MealSequence(object):
    def __init__(self, nomen,user,meal_list):
        self.nomen = nomen
        self.user = user
        self.meal_list = meal_list
        self.codgr_name_list = [x.codgr_name for x in self.meal_list]
        self.codsougr_name_list = [x.codsougr_name for x in self.meal_list]
        self.codal_name_list = [x.codal_name for x in self.meal_list]
        self.substitutions = []
        self.pandiet0 = 0
        self.current_pandiet = 0
        self.possibleSubstitutions = []
    
    
    def length(self):
        return sum([len(m.meal) for m in self.meal_list])
    
   
    def findMealIndex(self, day, tyrep):
        return next(i for i,m in enumerate(self.meal_list) 
                    if (m.tyrep == tyrep) & (m.day == day))
        
    def substituteItem(self, substitutionObject, level):
        day = substitutionObject.jour
        tyrep = substitutionObject.tyrep
        index = self.findMealIndex(day, tyrep)
        m = self.meal_list[index]
        
        level = substitutionObject.level
        x = substitutionObject.toSubstitute
        y = substitutionObject.substitutedBy
        indexToDelete = m.findItemIndex(level, getattr(x, level))
        m.meal.pop(indexToDelete)
        m.meal.append(y)
    
    def addToSubstitutionTrack(self, substitutionObject):
        self.substitutions.append(substitutionObject)
    
    def addToPossibleSubstitutions(self, substitutionList):
        self.possibleSubstitutions.append(substitutionList)
    
        
        
################################## FUNCTIONS
def fill_sequence_empty_meals(I):
    """
    Some meals are missing because they are not consumed. So each missing meal 
    is replaced by an empty meal
    """
    timestamps = [(i,j) for i in range(1,8) for j in range(1,7)]

    for i,seq in I.items():
        if len(seq) < 42:
            timestamp = [(m.day, m.tyrep) for m in seq]
            missing_timestamp = sorted(list(set(timestamps) - set(timestamp)), key = lambda x:(x[0], x[1]))  
            time_dict = {t:i for t, i in zip(timestamps, range(len(timestamps)))} 

            for t in missing_timestamp:
                meal = Meal([], t[1], t[0])
                index = time_dict[t]
                seq.insert(index,meal)
    return I


def createUserTable(df):
    """
    Read the user table and create the objects.
    
    Input :
        df (pd.DataFrame)
    Output : 
        dict (key nomen, value objects)
    """
    df.menopaus.fillna(0, inplace=True)
    df.regmaig.fillna(0, inplace=True)
    df.regmaig.fillna(0, inplace=True)
    df.regmedic.fillna(0, inplace=True)
    df.regrelig.fillna(0, inplace=True)
    df.regvegr.fillna(0, inplace=True)
    df.regvegt.fillna(0, inplace=True)
    df.bonalim.fillna(0, inplace=True)
    df.agglo9.fillna(0, inplace=True)
    
    U = {}
    for index, row in df.iterrows(): 
        u = User(index, row.sexe_ps, row.v2_age, row.poidsd, row.poidsm, row.bmi, 
             row.menopaus, row.regmaig, row.regmedic, row.regrelig, row.regvegr, 
             row.regvegt, row.ordi, row.bonalim, row.agglo9)
        U[index] = u
    return U
    
        
def get_consumption_sequence(df1, df_rep, users):
    """
    Read the table of consumption and create objects.
    
    Input :
        df (pd.DataFrame) consumption dataframe
        df_rep (pd.DataFrame) meal context dataframe
        users (dict) of User objects
    Output : 
        I_full 
        U (dict)
    """
    nomen_list = df1.nomen.unique().tolist()
    
    
    I = {n:[] for n in nomen_list} # Contient les séquences de consommation des individus
    M = {} #Contient les aliments d'un repas
    idx = pd.IndexSlice
    
    nomen = df1.nomen.iloc[0]
    jour = df1.jour.iloc[0]
    tyrep = df1.tyrep.iloc[0]
    
    for index, row in df1.iterrows(): # Scan tous les aliments de consommation
        #print(row.nomen, row.jour, row.tyrep)
        if (row.nomen == nomen) & (row.jour == jour) & (row.tyrep == tyrep):
            name = str(row.nomen) + '_' + str(index)
            M[name] = Food(row.codgr, row.codsougr, row.codal, row.codgr_name,
             row.codsougr_name, row.codal_name, row.qte_brute, row.qte_nette, 
             row.typal2, row.typal3) # Crée l'objet FoodItem

        else:
            m = list(set(M.values())) #Set of food items in a meal

            context = df_rep.loc[idx[nomen, jour, tyrep]]        
            meal = Meal(m, tyrep, jour, context.lieu, context.avecqui, 
                        context.duree)
            #print(meal)

            I[nomen].append(meal)

            M = {}
            nomen = row.nomen
            jour = row.jour
            tyrep = row.tyrep
            
    I_full = fill_sequence_empty_meals(I)
    U = {}
    for u,seq in I_full.items():
        user = users[u]
        u_seq = MealSequence(u, user, seq)
        U[u] = u_seq
    
    return I_full,U

def get_all_meals(I, cod, context = False):
    """
    Get all meals from the datatset with the 'cod' hierarchy.
    
    Input : 
        U : dict of MealSequence objects
        cod (str) in ['codgr', 'codsougr', 'codal']
    Ourput :
        meals : list of lists
    """
    meals = []
    if cod in ['codgr', 'codsougr', 'codal']:
        cod += '_name'
        print(cod)
        for m_u in I.values():
            for u in m_u:
                meal = getattr(u, cod)
                if meal:
                    c = attrgetter('tyrep', 'day', 'place', 'companion', 'duration')(u)
                    meal = [m for m in meal if m]
                    meals.append([c, meal])
    return meals
   
         
def main():
    pass


if __name__ == '__main__':
    main()
    

#def main():
#    """
#    cod in ['codsougr', 'codal', 'codgr']
#    """
#    #import data
#    #adultes = pd.read_pickle('adultes.p')
#    
#    path = os.path.join(os.getcwd(), 'data')
#    os.chdir(path) 
#    conso_ad = pd.read_pickle('conso_ad.p')
#    repas = pd.read_pickle('meal_context_clean.p') 
#    conso = conso_ad.reset_index()
#    
##    with open('dict_cod.p', 'rb') as handle:
##        dict_cod = pickle.load(handle)
#    
#    #process data
#    I,U = get_consumption_sequence(conso, repas)
#    
#    return I,U 

#I,U = main()
#
#with open('conso_seq_full.p', 'wb') as f:
#    pickle.dump(U, f)
#
#with open('conso_list_full.p', 'wb') as f:
#    pickle.dump(I, f)