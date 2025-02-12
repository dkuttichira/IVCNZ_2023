# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 08:46:49 2023

@author: deept
"""

import numpy as np
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train_check=y_train
y_test_check=y_test
y_train_check=y_train
x_train_feature=np.load(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10train_200relu.npy')
x_test_feature=np.load(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10test_200relu.npy')
class_percent_feature=np.load("class_percent_feature_threshmedian.npy")
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
mask_matrix = np.where(class_percent_feature > 10, 1, 0)
output_dim=10
input_dim=200
optimizer='adam'
epochs=5
x_train_masked=x_train_feature

for i in range(0,10):
    print(i)
    indexes=np.where(y_train_check==i)[0]
    x_train_masked[indexes]=np.multiply(x_train_masked[indexes],mask_matrix[i,:])
    indexes=[]
    
    
    
# def classification_model():
#     model = Sequential()
#     model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model

def classification_model():
    model = Sequential()
    model.add(Dense(25, input_dim = 200, activation= 'relu'))
    #model.add(Dense(64, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

keras_model = classification_model()
keras_model.fit(x_train_feature, y_train, epochs=epochs, verbose=1)

prediction_test=keras_model.predict(x_test_feature)
prediction_class_test=keras_model.predict_classes(x_test_feature)

accuracy_test=accuracy_score(prediction_class_test, y_test_check)
first_layer_biases  = keras_model.layers[0].get_weights()[0]