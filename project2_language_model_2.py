# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 17:29:32 2019

@author: fourn
"""



######################################
# Without Vectorization
    


from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
import keras.utils as ku
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from sklearn import cross_validation
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
    #return text[0:1000]


def dataset_preparation(data): 
    # Get list of phrases : ['NN FF EE TT', 'FF SS LL'...]
    phrases = data.split("\n")
    print('Number phrases : ', len(phrases))
    
    # Put the different chars of data in a list
    chars = sorted(set(' '.join(phrases).split(" ")))    # ['EE', 'FF', 'LL', 'NN', 'SS', 'TT']
    total_words = len(chars)
    char_indices = dict((c, i) for i, c in enumerate(chars))    # ['EE' : 0, 'FF' : 1, 'LL' : 2, 'NN' : 3, 'SS' : 4, 'TT' : 5]
    
    total_words = len(char_indices)
    
    # Get list of list of sequence of phrases : [[3, 0, 1, 5], [0, 4, 2]...]
    input_sequences = []
    for line in phrases:
        input_sequences.append(text_to_sequence_perso(line.split(" "), char_indices))
    print('Total input sequences : ', len(input_sequences))
    
    return input_sequences, total_words, char_indices, chars

def text_to_sequence_perso(line, char_indices):
    temp = []
    for word in line:
        temp.append(char_indices[word])
    return temp

def predictors_label(input_sequences, maxlen, total_words, type_predictor):
    if(type_predictor == 1):
        return predictors_label_1(input_sequences, maxlen, total_words)
    elif(type_predictor == 2):
        return predictors_label_2(input_sequences, maxlen, total_words)

def predictors_label_1(input_sequences, maxlen, total_words):
    # Get the predictors and the label for each phrases : label at the end of the inpu_sequence
        # Predictors :  [[3 0], [0 1], [0 4]]
       # Label : [[0 1 0 0 0 0], [0 0 0 0 0 1], [0 0 1 0 0 0]
       
    predictors = []    
    label = []   
    for i in range(0, len(input_sequences)):
        for j in range(0, len(input_sequences[i]) - maxlen):
            predictors.append(input_sequences[i][j: j + maxlen])
            label.append(input_sequences[i][j + maxlen])
    
    predictors = np.array(pad_sequences(predictors, maxlen=maxlen, padding='pre'))
    label = ku.to_categorical(label, num_classes=total_words)
    
    return predictors, label


def predictors_label_2(input_sequences, maxlen, total_words):
    # Get the predictors and the label for each phrases : label in the middle of the input_sequence
        # Predictors :  [[3 0], [0 1], [0 4]]
       # Label : [[0 1 0 0 0 0], [0 0 0 0 0 1], [0 0 1 0 0 0]
    
    middle = int(maxlen/2)
    predictors = []  
    label = []   
    for i in range(0, len(input_sequences)):
        for j in range(0, len(input_sequences[i]) - maxlen):
            predictors.append(input_sequences[i][j: j + middle] + input_sequences[i][j + middle + 1: j + maxlen + 1])
            label.append(input_sequences[i][j + middle])
    
    predictors = np.array(pad_sequences(predictors, maxlen=maxlen, padding='pre'))
    label = ku.to_categorical(label, num_classes=total_words)
    
    return predictors, label


def train_valid(x, y):
    x_train, x_valid, y_train, y_valid = cross_validation.train_test_split(x, y, test_size=0.2)
        
    return x_train, x_valid, y_train, y_valid


def create_model(x_train, y_train, x_valid, y_valid, maxlen, total_words):
    # build the model: a single LSTM
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=maxlen))
    model.add(LSTM(12))
    model.add(Dropout(0.2))
    model.add(Dense(total_words, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    model.fit(x_train, y_train, batch_size=128, epochs=50, verbose=1, callbacks=[earlystop], validation_data=(x_valid, y_valid))
    
    return model

def predict_char(model, predictors, labels, maxlen, chars):
        
    preds = model.predict(predictors, verbose=0)
    char_preds = []   # Character predicted : ['AA', 'BB',...]
    real_estimation =  []   # Estimation of the real character : [0.9, 0.21,... ]
    success = []   # 1 if good prediction, 0 else : [1, 0, 1, 1, ...]
    for i in range(len(preds)):
        pred = preds[i]
        index_max = np.argmax(pred)
        index_label = np.argmax(labels[i])
        char_preds.append(chars[index_max])
        real_estimation.append(pred[index_label])
        if(chars[index_max] == chars[index_label]):
            success.append(1)
        else:
            success.append(0)
    
    return char_preds, real_estimation, success


def principal():
    maxlen = utils.maxlen
    
    type_predictor = utils.type_predictor
    
    print('Data Load...')
    data = data_load('Corpus/CNN1.txt')
    print('Size of data : ', sys.getsizeof(data))
    
    print('Dataset preparation...')
    input_sequences, total_words, char_indices, chars = dataset_preparation(data)
    print('Len input sequence : ', len(input_sequences))
    print('Total words : ', total_words)
    del data
    gc.collect()
    utils.save_json(char_indices, 'data/char_indice_cnn_2')
    utils.save_json(chars, 'data/chars_cnn_2')
    
    print("Input sequence : ", input_sequences[0])
    
    
    print('Predictor and Label...')
    predictors, label = predictors_label(input_sequences, maxlen, total_words, type_predictor)
    del input_sequences
    gc.collect()
    
    print('Number predictor : ', len(predictors))
    print('Number label : ', len(label))
   
    print("Predictors : ", predictors[0])
    print("Label : ", label[0])
    
    print("Train Valid...")
    x_train, x_valid, y_train, y_valid = train_valid(predictors,label)
    del predictors, label
    gc.collect()
    
    print("Len x train : ", len(x_train))
    print("Len y train : ", len(y_train))
    print("Len x valid : ", len(x_valid))
    print("Len y valid : ", len(y_valid))
    
    
    print('Build model...')
    model = create_model(x_train, y_train, x_valid, y_valid, maxlen, total_words)
    del x_train, y_train, x_valid, y_valid, total_words
    gc.collect()
    
    
    model.save('data/model_language_cnn_2.h5')
    print('Model saved')
    del model
    gc.collect()
    

#principal()

