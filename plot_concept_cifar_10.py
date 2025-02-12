# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 00:59:12 2023

@author: deept
"""

import numpy as np
import matplotlib.pyplot as plt


mean_feature=np.load("concept_mean_cifar10.npy")
median_feature=np.load("concept_median_cifar10.npy")
#mode_feature=np.load("concept_mode_cifar10.npy")

def plot_concept_graph(x):
    plt.figure(figsize=(13,7))
    for i in range(0,x.shape[0]):
        my_label = "class_"+str(i)
        ticks=np.arange(0,x.shape[1])
        #plt.plot(ticks, x[i,:], label=my_label)
        plt.plot(ticks, x[i,:], marker='.',linewidth=4, markersize=13,label=my_label)
       
        my_label="_nolegend_"
        plt.plot(label="class_"+str(i))
    plt.xticks(np.arange(min(ticks), max(ticks)+1, 10),fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('neurons',fontsize=23)
    plt.ylabel('median_value',fontsize=23)
    #plt.title('Two lines on same graph!')
    plt.legend(fontsize=15,loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend(fontsize=15)
    plt.savefig("mediancifar10.png")
    plt.savefig("mediancifar10.pdf")
    
plot_concept_graph(median_feature)