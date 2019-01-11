# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 00:05:50 2018

@author: fourn
"""

import json

maxlen = 7
type_predictor = 2


def save_json(data, name):
    with open(name + '.json', 'w') as outfile:
        json.dump(data, outfile)
        
def load_json(name):
    with open(name+'.json') as json_file:  
        return json.load(json_file)
    
def remove_in_list(thing, my_list):
    while thing in my_list: my_list.remove(thing)
    
    
def same_chars(chars, chars2):
    same = 0
    notsame = []
    for i in range(len(chars2)):
        for j in range(len(chars)):
            if(chars2[i] == chars[j]):
                same = 1
        if(same == 0):
            notsame.append(chars2[i])
        same = 0
    
    print(notsame)
    

def average(lst):
    return sum(lst) / len(lst) 

def save_text(lst, path):
    F = open(path,"w") 
    for l in lst:
        F.write(str(l))
        F.write('\n')
    F.close()
    