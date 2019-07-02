# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 22:57:27 2019

@author: polo
"""

import json
from ImgurComments import get_Replies

DataSet = '/home/polo/.config/spyder-py3/PhD/Tourism30'

def Load_GalLery_Textual_Data(Country,gallery_id):
    with open(DataSet+'/'+Country+'/'+gallery_id+'/'+gallery_id+'.json') as data_file:    
         Data = json.load(data_file)
         S = []
         S.append(Data['title'])
         S.append([x['comment'] for x in Data['Comments']])
         S.append([get_Replies(x['children']) for x in Data['Comments'] if get_Replies(x['children'])])

    return S,Data

def get_Replies_of_Comments(children,Dict):
    if len(children) > 0:
       for child in children:
           for x in child['children']:
               if x['author_id'] != 0:
                  if x['author_id'] in Dict.keys():
                     Dict[x['author_id']].append(x['comment'])
                  else:
                      Dict[x['author_id']] = [x['comment']]
              
               Dict = get_Replies_of_Comments(x['children'],Dict)
    return Dict
           
def Load_GalLery_Comments(Country,gallery_id):
    with open(DataSet+'/'+Country+'/'+gallery_id+'/'+gallery_id+'.json') as data_file:    
         Data = json.load(data_file)
         Dict = {}
         
         for x in Data['Comments']:
             if x['author_id'] in Dict.keys():
                Dict[x['author_id']].append(x['comment'])
             else:
                 Dict[x['author_id']] = [x['comment']]
             Dict = get_Replies_of_Comments(x['children'],Dict)
    return Dict,Data

def Load_GoogleVision_Labels(Country,gallery_id):
    with open(DataSet+'/'+Country+'/'+gallery_id+'/google.json') as data_file:    
         Data = json.load(data_file)
         S = []
         if 'labelAnnotations' in Data.keys():
             #S.append([{'label':x['description'],'score':x['score']} for x in Data['labelAnnotations']])
             #S[0].extend([{'label':x['description'],'score':x['score'] } for x in Data['webDetection']['webEntities'] if ('description','score') in x.keys()])

             S = [x['description'] for x in Data['labelAnnotations']]
             S.extend([x['description'] for x in Data['webDetection']['webEntities'] if 'description' in x.keys()])
         else:
             #S.append([{'label':x['description'],'score':x['score'] } for x in Data['webDetection']['webEntities'] if 'description' in x.keys()])
             S = [x['description'] for x in Data['webDetection']['webEntities'] if 'description' in x.keys()]
         S.append(Data['webDetection']['bestGuessLabels'][0]['label'])
    return S,Data

#S ,Data  = Load_GalLery_Textual_Data('Algeria','x6TwpSQ')
#S1,Data1 = Load_GoogleVision_Labels('Algeria','x6TwpSQ')

L = ['x6TwpSQ','R3hT2v0','ZTGHc','U6LOSWO','AjChT','0fGGq','ziaXx','joMyrTQ','6aCY1be','DDaTZ1s']
for i in L:
    Dict,Data = Load_GalLery_Comments('Algeria',i)
    #print (i,len(Dict.keys()))
