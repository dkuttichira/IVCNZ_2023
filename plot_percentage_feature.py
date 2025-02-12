# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:03:39 2023

@author: deept
"""

import matplotlib.pyplot as plt
import numpy as np

data=np.load("class_percent_feature.npy")
b = np.where(data < 50, 0, data)

def plot(x,c):
    plt.figure(figsize=(13,7))
    
    neurons = np.arange(0,x.shape[0])
    activations = x
    plt.bar(neurons,activations)
    plt.xlabel('neurons',fontsize=23)
    plt.ylabel('activation_rate',fontsize=23)
    plt.savefig("p50class_"+str(c)+".png")
    plt.show()
    
for i in range(0,b.shape[0]):
    plot(b[i,:],i)