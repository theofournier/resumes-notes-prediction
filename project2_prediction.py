# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 00:03:19 2018

@author: fourn
"""
import project2_language_model as lm
import project2_resume_model as rm
import project2_utils as utils

from keras.models import load_model
import pickle
import io
import os

def data_load(path):
    file = io.open(path, encoding='utf-8')
    text = file.read()
    file.close()
    return text

def dataset_preparation(data): 
    # Get list of phrases : ['NN FF EE TT', 'FF SS LL'...]
    phrases = data.split("\n")
    print('Number phrases : ', len(phrases))
    
    # Get list of list of list of resume : [[['NN', 'FF', 'EE', 'TT'], ['FF', 'SS', 'LL']],...]]
    len_resumes = 0
    resumes = []
    notes = []
    temp = []
    column1 = ''
    column2 = ''
    note = 0
    if(len(phrases) > 0):
        phrase = phrases[0].split(" ")
        utils.remove_in_list("...", phrase)
        column1 = phrase[0]
        column2 = phrase[1]
        note = phrase[2]
        temp.append(phrase[3:])
        for i in range(len(phrases)):
            phrase = phrases[i].split(" ")
            utils.remove_in_list("...", phrase)
            if(len(phrase)>3):
                if(phrase[0] != column1 or phrase[1] != column2):
                    column1 = phrase[0]
                    column2 = phrase[1]
                    len_resumes += 1
                    notes.append(note)
                    resumes.append(temp)
                    temp = []
                    temp.append(phrase[3:])
                else:
                    note = phrase[2]
                    temp.append(phrase[3:])
        
    notes = list(map(int, notes))
    
    return resumes, notes

def accuracy_models(notes, preds):
    len_notes = len(notes)
    
    success = []
    
    for i in range(len_notes):
        if(notes[i] == preds[i]):
            success.append(1)
        else:
            success.append(0)
            
    return success

def principal():
    print('Load model...')
    exists = os.path.isfile('data/model_language_cnn.h5')
    if exists == False:
        lm.principal()
    model_language = load_model("data/model_language_cnn.h5")
    chars = utils.load_json('data/chars_cnn')
    char_indices = utils.load_json('data/char_indice_cnn')
    
    exists = os.path.isfile('data/model_resume.sav')
    if exists == False:
        rm.principal()
    model_resume = pickle.load(open("data/model_resume.sav", 'rb'))
    
    maxlen = utils.maxlen
    
    print('Data Load...')
    data = data_load('Corpus/input_learning_gramm_ST_2006.txt')
    
    print('Dataset preparation...')
    resumes, notes = dataset_preparation(data)
    del data
    
    print('Total resume : ', len(resumes))
    print('Total notes : ', len(notes))
    
    
    print('Predict notes...')
    preds = rm.predict_notes(model_resume, resumes, maxlen, model_language, chars, char_indices)
    del model_resume, model_language, resumes, maxlen, chars, char_indices
    
    print('Len notes : ', len(notes))
    print('Len preds : ', len(preds))
    
    success = accuracy_models(notes, preds)
    
    print('Nb success : ', sum(success))
    print('Accuracy : ', sum(success)/len(notes))
    
principal()