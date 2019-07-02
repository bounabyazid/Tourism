#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:16:44 2019

@author: polo
"""
import json
import spacy
#https://spacy.io/usage/linguistic-features#named-entities
import numpy as np
from fuzzywuzzy import fuzz

from LoadTextData import Load_GalLery_Comments, Load_GoogleVision_Labels
from ImgurComments import Countries, galeries

nlp = spacy.load("en_core_web_sm")

Country_galeries = np.array(galeries).reshape(len(Countries),10).tolist()

def Hypothesis_1():
    #mentioning nearby locations in Comments
    Stats = {}
    CountDictLocs = {}
    for Country in Countries:
        GalleryLocs = {}
        galeries = []
        for gallery in Country_galeries[Countries.index(Country)]:
            galeries
            Dict,Data = Load_GalLery_Comments(Country,gallery)
            UserLocs = {}
            N = 0
            for key in Dict.keys():
                N += len(Dict[key])
                for comment in Dict[key]:
                    doc = nlp(comment)
                    for ent in doc.ents:
                        if ent.label_ == 'LOC':#in ['FAC', 'GPE', 'LOC']:
                           if key in UserLocs.keys():
                              UserLocs[key].append(ent.text)
                           else:
                               UserLocs[key] = [ent.text]
            GalleryLocs[gallery] = UserLocs
            galeries.append([N])
        CountDictLocs[Country] = GalleryLocs
        
        Stats[Country] = galeries
    
    with open('Stats.json','w') as f:
         json.dump(Results,f)
    
    with open('CommentsLocs.json','w') as f:
         json.dump(CountDictLocs,f)
    return CountDictLocs

with open('CommentsLocs.json') as data_file:    
     CountDictLocs = json.load(data_file)
    
with open('Stats.json') as data_file:    
     Stats = json.load(data_file)
     
def Sim_Hypothesis_1():
    #CountDictLocs = Hypothesis_1()
    Similarities = {}
    for Country in CountDictLocs.keys():
        Similarities[Country] = {k:0 for k in CountDictLocs[Country].keys()}
        i = 0
        for gallery in CountDictLocs[Country].keys():
            Locs = []
            N = len(CountDictLocs[Country][gallery].keys()) 
            if N != 0:
               for user in CountDictLocs[Country][gallery].keys():
                   Locs.extend(CountDictLocs[Country][gallery][user])
                   
                   #the problem is when user mention two loctions we would have 
                   #user with two or more comments so should we consider one location or 
                   #both locations counted in the next formulla
               
               Similarities[Country][gallery] = round(len(list(set(Locs)))/N, 2)
            if N == 0:
               Stats[Country][i].extend([len(list(set(Locs))), N, 0])
            else:
                Stats[Country][i].extend([len(list(set(Locs))), N, round(len(list(set(Locs)))/N, 2)*100])
            
            i+=1       
            #if Country == 'Algeria':
            #   if Similarities[Country][gallery] != 0:
            #      print (gallery, len(list(set(Locs))), N)
            #   else:
            #       print (gallery, len(list(set(Locs))), N)
               
    return CountDictLocs, Similarities,Stats
      
def Hypothesis_2():
    Similarities = {}
    for Country in Countries:
        Similarities[Country] = {k:0 for k in CountDictLocs[Country].keys()}
        for gallery in Country_galeries[Countries.index(Country)]:
            Dict,Data = Load_GalLery_Comments(Country,gallery)
            setB, entities = Load_GoogleVision_Labels(Country,gallery)
            
            overlap = 0

            setA = []
            
            for item in list(Dict.values()):
                setA.extend(item)
            
            setA = list(set(setA))
            
            for A in setA:
                for B in setB:
                    if fuzz.ratio(A, B) >= 75:
                       overlap += 1
                       break
            
            uni = list(set(setA) | set(setB))
            universe = [uni[0]]

            for i in range(len(uni)):
                for j in range(i+1,len(uni)):
                    if fuzz.ratio(uni[i], uni[j]) < 75 and uni[j] not in universe:
                       universe.append(uni[j])
               
            universe = len(universe)

            overall = round(float(overlap) / float(universe) * 100., 2)
            #print ('overlap = ',overlap)
            #print ('universe = ',universe)
            #print (overall)

            Similarities[Country][gallery] = overall
    return Similarities       

def Mean_Galeries():
    for Country in Countries:
        for i in Stats[Country]:
            print(i)
        break    
        
#Hypothesis_1()
    
#CountDictLocs, Similarities,Stats = Sim_Hypothesis_1()
#Similarities2 = Hypothesis_2()
#Dict,Data = Load_GalLery_Comments('Algeria','x6TwpSQ')
    
Mean_Galeries()
