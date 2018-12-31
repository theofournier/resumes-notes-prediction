# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 22:55:41 2018

@author: fourn
"""

import project2_language_model as lm
import project2_utils as utils

import numpy as np

from keras.models import load_model
import pickle
from sklearn import cross_validation, svm
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

def stat(real_estimation, success):
    stat = []
    stat.append(max(real_estimation))
    stat.append(min(real_estimation))
    stat.append(sum(real_estimation))
    stat.append(np.mean(real_estimation))
    stat.append(np.median(real_estimation))
    stat.append(np.std(real_estimation))
    stat.append(sum(success))
    stat.append(np.mean(success))
    stat.append(np.median(success))
    stat.append(np.std(success))
    
    return stat
        


def get_stats(resumes, maxlen, model, chars, char_indices):
    # For each resume, get the prediction
    stats = []   # 1 line per resume with different stats : list of list
    for i in range(len(resumes)):
        predictors, labels = lm.predictors_label(resumes[i], maxlen)
        char_preds, real_estimation, success = lm.predict_char(model, predictors, labels, maxlen, chars, char_indices)
        stats.append(stat(real_estimation, success))
        
    return stats

def train_valid(x, y):
    x_train, x_valid, y_train, y_valid = cross_validation.train_test_split(x, y, test_size=0.2)
        
    return x_train, x_valid, y_train, y_valid


def create_model(x_train, y_train, x_valid, y_valid):
    # build the model: a single SVM
    model = svm.SVC()

    model.fit(x_train, y_train)
    print('Accuracy of SVM classifier on training set: {:.2f}'.format(model.score(x_train, y_train)))
    print('Accuracy of SVM classifier on test set: {:.2f}'.format(model.score(x_valid, y_valid)))
    return model


def predict_notes(model, resumes, maxlen, model_language, chars, char_indices):
    stats = get_stats(resumes, maxlen, model_language, chars, char_indices)
    return model.predict(stats)


def principal():
    exists = os.path.isfile('data/model_language_cnn.h5')
    if exists == False:
        lm.principal()
        
    model_language = load_model("data/model_language_cnn.h5")
    chars = utils.load_json('data/chars_cnn')
    char_indices = utils.load_json('data/char_indice_cnn')
    
    maxlen = utils.maxlen
    
    print('Data Load...')
    data = data_load('Corpus/input_learning_gramm_ST_2007.txt')
    
    print('Dataset preparation...')
    resumes, notes = dataset_preparation(data)
    del data
    
    print('Total resume : ', len(resumes))
    print('Total notes : ', len(notes))
    
    print('Get stats...')
    stats = get_stats(resumes, maxlen, model_language, chars, char_indices)
    del resumes, model_language, chars, char_indices
    
    print('Number of stats : ', len(stats))
    
    print("Train Valid...")
    x_train, x_valid, y_train, y_valid= train_valid(stats, notes)
    del notes, stats
    
    print('Len x_train : ', len(x_train))
    print('Len y_train : ', len(y_train))
    print('Len x_valid : ', len(x_valid))
    print('Len y_valid : ', len(y_valid))
    
    
    print('Build model...')
    model = create_model(x_train, y_train, x_valid, y_valid)
    del x_train, y_train, x_valid, y_valid
    
    pickle.dump(model, open('data/model_resume.sav', 'wb'))
    print('Model saved')
    del model


#principal()


