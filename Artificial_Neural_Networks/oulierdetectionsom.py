# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:32:05 2019

@author: Mahamed
"""
import sys

from minisom import MiniSom

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 


from sklearn.datasets import make_blobs
from sklearn.preprocessing import scale

outliers_percentage = 0.35
inliers = 300
outliers = int(inliers * outliers_percentage)
data = make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[.3, .3],n_samples=inliers, random_state=0)[0]
data = scale(data)
data = np.concatenate([data, (np.random.rand(outliers, 2)-.5)*4.])

#from sklearn.datasets import make_circles
#data = make_circles(noise=.1, n_samples=inliers, random_state=0)[0]
#data = scale(data)
#data = np.concatenate([data, 
#                       (np.random.rand(outliers, 2)-.5)*4.])
#
#plt.figure(figsize=(8, 8))
#plt.scatter(data[:, 0], data[:, 1],
#            label='data')
#plt.plot(data[:,0],data[:,1], 'ro', alpha = 0.5)
#for i in range(data.shape[0]):
#    plt.text(data[i,0], data[i,1], str(i))
#
#plt.show()


som = MiniSom(2, 1, data.shape[1], sigma=1, learning_rate=0.5, neighborhood_function='triangle', random_seed=10)
som.train_batch(data, 100)

quantization_errors = np.linalg.norm(som.quantization(data) - data, axis=1)
error_treshold = np.percentile(quantization_errors, 100*(1-outliers_percentage)+5)

     
is_outlier = quantization_errors > error_treshold
plt.hist(quantization_errors)
plt.axvline(error_treshold, color='k', linestyle='--')
plt.xlabel('error')
plt.ylabel('frequency')
#le quantization est mesure avec le error treshold si la valeur est superieur on l'appel oulier



plt.figure(figsize=(8, 8))
plt.scatter(data[~is_outlier, 0], data[~is_outlier, 1],
            label='inlier')
plt.scatter(data[is_outlier, 0], data[is_outlier, 1],
            label='outlier')
plt.legend()
plt.savefig('resulting_images/som_outliers_detection.png')
plt.show()
valuesoutlier=data[is_outlier]
datawithoutoutlier=data[~is_outlier]
#https://github.com/JustGlowing/minisom/blob/master/examples/OutliersDetection.ipynb