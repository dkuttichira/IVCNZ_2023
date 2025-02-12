# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:35:20 2023

@author: deept
"""

import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from scipy import stats
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train= (x_train.astype('float32'))/255
x_test= (x_test.astype('float32'))/255
model=load_model("cifar10model_200.h5")
x_trainm= np.load(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10train_200relu.npy')
#x_testm= np.load(r'C:\Users\deept\.spyder-py3\CNN_feature\cifar10_dbr\cifar10test_200.npy')
val=np.amax(x_trainm)
x_trainm=x_trainm/val

#x_trainm[x_trainm<0] = 0
#x_train_feature_modified=np.round(x_trainm,2)
min_val_arr=[]


def correct_classified_instances(x,y):
    predicted=model.predict_classes(x)
    diff_predicted=predicted-y.ravel()
    indexes=np.where(diff_predicted==0)[0]
    return indexes

def class_specific_instances(x,y,c):
    indexes=np.where(corr_classified_labels==c)[0]
    return x[indexes,:]

corr_classified_instance_index=correct_classified_instances(x_train,y_train)
corr_classified_instances=x_trainm[corr_classified_instance_index,:]
corr_classified_labels=y_train[corr_classified_instance_index]#.reshape(len(corr_classified_instance_index),1)
#corr_data=np.hstack([corr_classified_instances,corr_classified_labels])
for i in range(0,10):
    corr_class_specific_instances=class_specific_instances(corr_classified_instances,corr_classified_labels,i)
    lower_range=np.median(corr_class_specific_instances,axis=0)
    #lower_range=stats.mode(np.round(corr_class_specific_instances,2))[0].ravel()
    min_val_arr.append(lower_range)
    print(i)
min_val_arr=np.array(min_val_arr)
np.save("concept_median_cifar10.npy",min_val_arr)