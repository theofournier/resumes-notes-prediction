# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 10:13:54 2018

@author: fourn
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from sklearn import cross_validation
import numpy as np
import io
import project2_utils as utils


def data_load(path):
    file = io.open(path, encoding='utf-8')
    text = file.read()
    file.close()
#    return text[0:81436370]
    return text[0:10000]

def dataset_preparation(data): 
    # Get list of articles : ['NN FF EE TT', 'FF SS LL'...]
    phrases = data.split("\n")
    print('Number phrases : ', len(phrases))
    
    # Get list of list articles : [['NN', 'FF', 'EE', 'TT'], ['FF', 'SS', 'LL']...]
    input_sequences = []
    for i in range(0, len(phrases)):
        input_sequences.append(phrases[i].split(" "))
    print('Total input sequences : ', len(input_sequences))
    
        
    # Put the different chars of data in a list
    chars = sorted(set(' '.join(phrases).split(" ")))
    print('Chars : ', chars)
    total_words = len(chars)
    print('Total words : ', total_words)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    print('Char indice : ', char_indices)
    
    return input_sequences, chars, char_indices, total_words
    

def predictors_label(input_sequences, maxlen):
    # Get the predictors and the label for each phrases
    predictors = []
    label = []
    for i in range(0, len(input_sequences)):
        for j in range(0, len(input_sequences[i]) - maxlen):
            predictors.append(input_sequences[i][j: j + maxlen])
            label.append(input_sequences[i][j + maxlen])
    
    return predictors, label
    
def vectorization(predictors, label, maxlen, chars, char_indices):
    x = np.zeros((len(predictors), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(predictors), len(chars)), dtype=np.bool)
    for i, predictor in enumerate(predictors):
        for t, char in enumerate(predictor):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[label[i]]] = 1
    
    return x, y

def train_valid(x, y):
    x_train, x_valid, y_train, y_valid = cross_validation.train_test_split(x, y, test_size=0.2)
        
    return x_train, x_valid, y_train, y_valid


def create_model(x_train, y_train, x_valid, y_valid, maxlen, total_words):
    # build the model: a single LSTM
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, total_words)))
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
    
    print('Dataset preparation...')
    input_sequences, chars, char_indices, total_words = dataset_preparation(data)
    del data
    utils.save_json(chars, 'data/chars_cnn')
    utils.save_json(char_indices, 'data/char_indice_cnn')
    
    print('Predictor and Label...')
    predictors, label = predictors_label(input_sequences, maxlen)
    del input_sequences
    
    print('Number predictor : ', len(predictors))
    print('Number label : ', len(label))
    print('First predictor : ', predictors[0])
    print('First label : ', label[0])
    
    
    print('Vectorization...')
    x, y = vectorization(predictors, label, maxlen, chars, char_indices)
    del predictors, label
    
    print('Len X : ', len(x))
    print('Len Y : ', len(y))
    
    
    print("Train Valid...")
    x_train, x_valid, y_train, y_valid = train_valid(x,y)
    del x, y
    
    print(len(x_train))
    print(len(y_train))
    print(len(x_valid))
    print(len(y_valid))
    
    
    print('Build model...')
    model = create_model(x_train, y_train, x_valid, y_valid, maxlen, total_words)
    del x_train, y_train, x_valid, y_valid, total_words
    
    model.save('data/model_language_cnn.h5')
    print('Model saved')
    del model

#principal()
