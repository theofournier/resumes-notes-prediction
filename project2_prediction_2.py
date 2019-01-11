# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 00:03:19 2018

@author: fourn
"""
import project2_language_model_2 as lm
import project2_resume_model_2 as rm
import project2_utils as utils

from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import io
import os

from sklearn.externals import joblib

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
        for i in range(1, len(phrases)):
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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



def principal():
    print('Load model...')
    exists = os.path.isfile('data/model_language_cnn_2.h5')
    if exists == False:
        print('Language model generation')
        lm.principal()
    model_language = load_model("data/model_language_cnn_2.h5")
    chars = utils.load_json('data/chars_cnn_2')
    char_indices = utils.load_json('data/char_indice_cnn_2')
    
    exists = os.path.isfile('data/model_resume_2.sav')
    if exists == False:
        print('Resume model generation')
        rm.principal()
    model_resume = joblib.load("data/model_resume_2.sav")
    
    maxlen = utils.maxlen
    total_words = len(char_indices)
    type_predictor = 2
    
    print('Data Load...')
    data = data_load('Corpus/input_learning_gramm_ST_2006.txt')
    
    print('Dataset preparation...')
    resumes, notes = dataset_preparation(data)
    del data
    
    print('Total resume : ', len(resumes))
    print('Total notes : ', len(notes))
    
    
    print('Predict notes...')
    preds = rm.predict_notes(model_resume, resumes, maxlen, model_language, total_words, chars, char_indices, type_predictor)
    del model_resume, model_language, resumes, maxlen, chars, char_indices
    
    print('Len notes : ', len(notes))
    print('Len preds : ', len(preds))
    exists = os.path.isfile('notes_preds_2.txt')
    if(exists):
        os.remove("notes_preds_2.txt")
    utils.save_text(preds, "notes_preds_2.txt")
    
    success = accuracy_models(notes, preds)
    
    print('Nb success : ', sum(success))
    print('Accuracy : ', sum(success)/len(notes))
    
    class_names = ["1", "2","3","4","5"]
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(notes, preds)
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')
    
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    
    plt.show()
    
principal()