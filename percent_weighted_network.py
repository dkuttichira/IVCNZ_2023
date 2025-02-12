# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:42:09 2023

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
x_train_feature=np.load(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10train_200relu.npy')
x_test_feature=np.load(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10test_200relu.npy')
class_percent_feature=np.load("class_percent_feature_threshmedian.npy")
val=np.amax(x_train_feature)
x_test_feature=x_test_feature/val
weights=class_percent_feature/100
set_weights=np.zeros((10, 200))

for i in range(0,10):
    ind = np.argpartition(weights[i,:], -130)[-130:]
    np.put(set_weights[i,:], ind, weights[i,ind])
# set_weights= (np.where(weights < 0.5, 0, weights))
output_dim=10
input_dim=200

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
optimizer='adam'
set_weights=set_weights.transpose()

def classification_model():
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

#training
final_weights=[]
keras_model = classification_model()
first_layer_biases  = keras_model.layers[0].get_weights()[1]
final_weights.append(set_weights)
final_weights.append(first_layer_biases)
keras_model.layers[0].set_weights(final_weights) 
#keras_model.fit(x_train_feature, y_train, epochs=5, verbose=1)
prediction_test=keras_model.predict(x_test_feature)
prediction_class_test=keras_model.predict_classes(x_test_feature)

prediction_train=keras_model.predict(x_train_feature)
prediction_class_train=keras_model.predict_classes(x_train_feature)

accuracy_test=accuracy_score(prediction_class_test, y_test_check)
accuracy_train=accuracy_score(prediction_class_train, y_train_check)