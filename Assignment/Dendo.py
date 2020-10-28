# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:48:53 2020

@author: Samip
"""

import numpy as np

X = np.array([[4,7],
    [8,3],
    [10,2],
    [1,4],
    [2,2],
    [1,3],
    [1,2],
    [2,1],
    [10,3],
    [4,8],
    [5,7],
    [4,4],])

import matplotlib.pyplot as plt

labels = range(1, 11)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1], label='True Position')

for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked = linkage(X, 'single')

labelList = range(1, 13)

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()