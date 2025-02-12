# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:14:46 2023

@author: deept
"""

import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras.layers import Dense

seed=21
class_num=10
epochs=100
optimizer='adam'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_test_check=y_test
# x_train= (x_train.astype('float32'))/255
# x_test= (x_test.astype('float32'))/255

x_train= np.load(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10train_200.npy')
x_test= np.load(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10test_200.npy')
val=np.amax(x_train)
x_train=x_train/val
x_test=x_test/val


y_test_check=y_test
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
#x_train_concept_feature_min = np.load("x_train_concept_feature_min.npy")
x_train_concept_feature_mean = np.load("x_train_concept_feature_mean.npy")
x_train_concept_feature_max = np.load("x_train_concept_feature_max.npy")
x_train_concept_feature_median = np.load("x_train_concept_feature_median.npy")

#x_test_concept_feature_min = np.load("x_test_concept_feature_min.npy")
x_test_concept_feature_mean = np.load("x_test_concept_feature_mean.npy")
x_test_concept_feature_max = np.load("x_test_concept_feature_max.npy")
x_test_concept_feature_median = np.load("x_test_concept_feature_median.npy")

x_train_concept_feature= x_train_concept_feature_mean+x_train_concept_feature_max
x_test_concept_feature=x_test_concept_feature_mean+x_test_concept_feature_max

x_train=np.hstack((x_train_concept_feature_median,x_train_concept_feature_mean))
x_test=np.hstack((x_test_concept_feature_median,x_test_concept_feature_mean))
#model construction
input_dim = 20 # 4 variables
output_dim = 10 # 3 possible outputs
epochs=100

def classification_model():
    model = Sequential()
    model.add(Dense(output_dim, input_dim=input_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#training
keras_model = classification_model()
keras_model.fit(x_train, y_train, epochs=epochs, verbose=1)

prediction=keras_model.predict(x_test)
prediction_class=keras_model.predict_classes(x_test)

accuracy=accuracy_score(prediction_class, y_test_check)