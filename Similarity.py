#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:56:15 2019
@author: polo
"""
import json

import numpy as np
import matplotlib.pyplot as plt

from fuzzywuzzy import fuzz

from gensim.summarization import keywords

from LoadTextData import Load_GalLery_Textual_Data,Load_GoogleVision_Labels
from LDA import Preprocessing,PrepareData,LDA,Topics_Words

from ImgurComments import Countries,galeries
    
DataSet = '/home/polo/.config/spyder-py3/PhD/Tourism30'

#______________________________________________________________________________

def LoadTextData(Country,gallery_id):
    
    S ,Data  = Load_GalLery_Textual_Data(Country,gallery_id)
    
    S1 ,Data1  = Load_GoogleVision_Labels(Country,gallery_id)
    
    labels = [Preprocessing(x) for x in S1]
    
    DocList = S[1]
    DocList.append(S[0])

    for s in S[2]:
        DocList.extend(s)
    
    data_lemmatized = PrepareData(DocList)
    
    lda_model,id2word,corpus = LDA(data_lemmatized,num_topics=20)#len(labels))
    Topic_Words = Topics_Words(lda_model,num_words=len(labels))   
    
    return data_lemmatized,Topic_Words,labels

#______________________________________________________________________________
    
def FuzzyWazzy_Similarity(Country,gallery_id):
    'https://marcobonzanini.com/2015/02/25/fuzzy-string-matching-in-python/'
    data_lemmatized,Topic_Words,labels = LoadTextData(Country,gallery_id)

    setA = set(labels)
    i = 0
    for Topic in Topics:
        print ('____________Topic:{}_____________'.format(i))
        setB = set(Topic)
        
        overlap = 0
        for l in setA:
            for w in setB:
                if fuzz.ratio(l, w) >= 75:
                   overlap += 1
               
    
        uni = list(set(setA) | set(setB))
        universe = [uni[0]]

        for i in range(len(uni)):
            for j in range(i+1,len(uni)):
                if fuzz.ratio(uni[i], uni[j]) <= 75 and uni[j] not in universe:
                   universe.append(uni[j])
               
        universe = len(universe)

        labels = round(float(overlap) / len(setA) * 100., 2)
        comments = round(float(overlap) / len(setB) * 100., 2)
        overall = round(float(overlap) / float(universe) * 100., 2)
    
        print ('overlap = ',overlap)
        print ('universe = ',universe)
        
        print ('Labels = ',len(setA))
        print ('Comments = ',len(setB))

        print ('overlap(Labels,Comments)/Labels = ',labels)
        print ('overlap(Labels,Comments)/Comments = ',comments)
    
        print ('overlap(Labels,Comments)/Universe(Labels,Comments) = ',overall)
    
        i = i+1
        
    return Topics,setA

#______________________________________________________________________________

def FuzzyWazzy_SimilarityOverAll(Country,gallery_id):
    data_lemmatized,Topic_Words,labels = LoadTextData(Country,gallery_id)

    #print ('=============OverAll Similarity==============')
    
    setA = list(set(labels))
    setB = list(set([w for Topic in Topic_Words for w in Topic]))
    
    overlap = 0
    
    for l in setA:
        for w in setB:
            if fuzz.ratio(l, w) >= 80:
               overlap += 1
            
    universe = []
    
    uni = list(set(setA) | set(setB))
    
    universe.append(uni[0])    
    for i in range(len(uni)-1):
        for j in range(i+1,len(uni)):
            if fuzz.ratio(uni[i], uni[j]) < 80 and uni[j] not in universe:
               universe.append(uni[j])
               
    universe = len(universe)
    
    labels = round(float(overlap) / len(setA) * 100., 2)
    comments = round(float(overlap) / len(setB) * 100., 2)
    overall = round(float(overlap) / float(universe) * 100., 2)
        
    print ('overlap = ',overlap)
    print ('universe = ',universe)
    
    print ('len(Labels) = ',len(setA))
    print ('len(Comments) = ',len(setB))

    print ('overlap(Labels,Comments)/Labels = ',labels)
    print ('overlap(Labels,Comments)/Comments = ',comments)
    
    print ('overlap(Labels,Comments)/Universe(Labels,Comments) = ',overall)
    
    return labels,comments,overall

#______________________________________________________________________________
    
def keyWords_Labels_Matching(Country,gallery_id):
    data_lemmatized,Topic_Words,labels = LoadTextData(Country,gallery_id)
        
    Text = [w for doc in data_lemmatized for w in doc]
       
    fullStr = ' '.join(Text)
            
    setA = list(set(labels))
    
    setB = keywords(fullStr).split('\n')

    setB = [w for docs in PrepareData(setB) for w in docs]
  
    overlap = 0
    
    for l in setA:
        for w in setB:
            if fuzz.ratio(l, w) >= 80:
               overlap += 1
               
    universe = []
    
    uni = list(set(setA) | set(setB))
    
    universe.append(uni[0])    
    for i in range(len(uni)-1):
        for j in range(i+1,len(uni)):
            if fuzz.ratio(uni[i], uni[j]) < 80 and uni[j] not in universe:
               universe.append(uni[j])

               
    universe = len(universe)
    
    labels = round(float(overlap) / len(setA) * 100., 2)
    comments = round(float(overlap) / len(setB) * 100., 2)
    overall = round(float(overlap) / float(universe) * 100., 2)
        
    print ('overlap = ',overlap)
    print ('universe = ',universe)
    
    print ('len(Labels) = ',len(setA))
    print ('len(Comments) = ',len(setB))

    print ('overlap(Labels,Comments)/Labels = ',labels)
    print ('overlap(Labels,Comments)/Comments = ',comments)
    
    print ('overlap(Labels,Comments)/Universe(Labels,Comments) = ',overall)
    
    return labels,comments,overall#,setA,setB

#______________________________________________________________________________
       
def OverAll_Text_Similarity_DataSet():
   
    Galeries_Matrix = np.array(galeries).reshape(len(Countries),10)

    i = 0
    for Country in Countries:
        print ('============='+Country+'==============')

        Slabels = []
        Scomments = []
        Soverall = []
        
        Similarities = {}
        
        for j in range (10):
            #print(Galeries_Matrix[i,j])
            #labels,comments,overall = FuzzyWazzy_SimilarityOverAll(Country,Galeries_Matrix[i,j])
            labels,comments,overall = keyWords_Labels_Matching(Country,Galeries_Matrix[i,j])
            
            Slabels.append(labels)
            Scomments.append(comments)
            Soverall.append(overall)
        
        Similarities['labels'] = Slabels
        Similarities['comments'] = Scomments
        Similarities['overall'] = Soverall
        
        with open(DataSet+'/'+Country+'/Similarities.json', 'w') as outfile:
             json.dump(Similarities, outfile)
        #break             
        i+=1

#______________________________________________________________________________
        
def Histogramme(Country):
    with open(DataSet+'/'+Country+'/Similarities.json') as data_file:    
         Data = json.load(data_file)
    #plt.hist(Data['overall'])
    
    x = np.arange(10)
    plt.bar(x, Data['overall'])
    plt.xticks(x+.2, x)

#Topics,labels = FuzzyWazzy_Similarity('Algeria','x6TwpSQ')

#OverAll_Text_Similarity_DataSet()
Histogramme('Algeria')

#labels,comments,overall,setA,setB = keyWords_Labels_Matching('Algeria','x6TwpSQ')

#keyWords_Labels_Matching('Algeria','x6TwpSQ')

#data_lemmatized,Topic_Words,labels = LoadTextData('Algeria','x6TwpSQ')