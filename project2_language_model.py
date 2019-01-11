# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 10:13:54 2018

@author: fourn
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from sklearn.cross_validation import train_test_split
import numpy as np
import io
import project2_utils as utils
import gc
import sys


def data_load(path):
    file = io.open(path, encoding='utf-8')
    text = file.read()
    file.close()
    return text[0:3806020]
 #   return text

def dataset_preparation(data): 
    # Get list of phrases : ['NN FF EE TT', 'FF SS LL'...]
    phrases = data.split("\n")
    print('Number phrases : ', len(phrases))
    
    # Get list of list phrases : [['NN', 'FF', 'EE', 'TT'], ['FF', 'SS', 'LL']...]
    input_sequences = []
    for i in range(0, len(phrases)):
        input_sequences.append(phrases[i].split(" "))
    print('Total input sequences : ', len(input_sequences))
    
        
    # Put the different chars of data in a list
    chars = sorted(set(' '.join(phrases).split(" ")))    # ['EE', 'FF', 'LL', 'NN', 'SS', 'TT']
    print('Chars : ', chars)
    total_words = len(chars)
    print('Total words : ', total_words)
    char_indices = dict((c, i) for i, c in enumerate(chars))    # ['EE' : 0, 'FF' : 1, 'LL' : 2, 'NN' : 3, 'SS' : 4, 'TT' : 5]
    print('Char indice : ', char_indices)
    
    return input_sequences, chars, char_indices, total_words
    

def predictors_label(input_sequences, maxlen):
    # Get the predictors and the label for each phrases
    predictors = []    # [['NN', 'FF'], ['FF', 'EE'], ['FF', 'SS']]
    label = []    # ['EE', 'TT', 'LL']
    for i in range(0, len(input_sequences)):
        for j in range(0, len(input_sequences[i]) - maxlen):
            predictors.append(input_sequences[i][j: j + maxlen])
            label.append(input_sequences[i][j + maxlen])
    
    return predictors, label
    
def vectorization(predictors, label, maxlen, chars, char_indices):
    x = np.zeros((len(predictors), maxlen, len(chars)), dtype=np.bool)   # [[[0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0]], [[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]]
    y = np.zeros((len(predictors), len(chars)), dtype=np.bool)   # [[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0]]
    for i, predictor in enumerate(predictors):
        for t, char in enumerate(predictor):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[label[i]]] = 1
    
    return x, y

def train_valid(x, y):
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)
        
    return x_train, x_valid, y_train, y_valid


def create_model(x_train, y_train, x_valid, y_valid, maxlen, total_words):
    # build the model: a single LSTM
    model = Sequential()
    model.add(LSTM(12, input_shape=(maxlen, total_words)))
    model.add(Dense(total_words, activation='softmax'))
    
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    model.summary()
    
    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_valid, y_valid))
    
    return model

def predict_char(model, predictors, labels, maxlen, chars, char_indices):
    x, y = vectorization(predictors, labels, maxlen, chars, char_indices)
        
    preds = model.predict(x, verbose=0)
    char_preds = []   # Character predicted : ['AA', 'BB',...]
    real_estimation =  []   # Estimation of the real character : [0.9, 0.21,... ]
    success = []   # 1 if good prediction, 0 else : [1, 0, 1, 1, ...]
    for i in range(len(preds)):
        pred = preds[i]
        index_max = np.argmax(pred)
        char_preds.append(chars[index_max])
        real_estimation.append(pred[char_indices[labels[i]]])
        if(chars[index_max] == labels[i]):
            success.append(1)
        else:
            success.append(0)
    
    return char_preds, real_estimation, success
    
    
def principal():
    maxlen = utils.maxlen
    print('Data Load...')
    data = data_load('Corpus/CNN1.txt')
    print('Size of data : ', sys.getsizeof(data))
    
    print('Dataset preparation...')
    input_sequences, chars, char_indices, total_words = dataset_preparation(data)
    print('Size of input_sequences : ', sys.getsizeof(input_sequences))
    print('Size of chars : ', sys.getsizeof(chars))
    print('Size of char_indices : ', sys.getsizeof(char_indices))
    del data
    gc.collect()
    utils.save_json(chars, 'data/chars_cnn_2')
    utils.save_json(char_indices, 'data/char_indice_cnn_2')
    
    print('Predictor and Label...')
    predictors, label = predictors_label(input_sequences, maxlen)
    print('Size of predictors : ', sys.getsizeof(predictors))
    print('Size of label : ', sys.getsizeof(label))
    del input_sequences
    gc.collect()
    
    print('Number predictor : ', len(predictors))
    print('Number label : ', len(label))
    
    
    print('Vectorization...')
    x, y = vectorization(predictors, label, maxlen, chars, char_indices)
    print('Size of x : ', sys.getsizeof(x))
    print('Size of y : ', sys.getsizeof(y))
    del predictors, label
    gc.collect()
    
    print('Len X : ', len(x))
    print('Len Y : ', len(y))
    
    
    print("Train Valid...")
    x_train, x_valid, y_train, y_valid = train_valid(x,y)
    print('Size of x_train : ', sys.getsizeof(x_train))
    print('Size of x_valid : ', sys.getsizeof(x_valid))
    print('Size of y_train : ', sys.getsizeof(y_train))
    print('Size of y_valid : ', sys.getsizeof(y_valid))
    del x, y
    gc.collect()
    
    print(len(x_train))
    print(len(y_train))
    print(len(x_valid))
    print(len(y_valid))
    
    
    print('Build model...')
    model = create_model(x_train, y_train, x_valid, y_valid, maxlen, total_words)
    print('Size of model : ', sys.getsizeof(model))
    del x_train, y_train, x_valid, y_valid, total_words
    gc.collect()
    
    model.save('data/model_language_cnn.h5')
    print('Model saved')
    del model
    gc.collect()

#principal()
