# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:45:28 2020

@author: Samip
"""

import operator
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed

df = pd.read_csv(r"dataset.csv")

x=round(df['X'].mean())

s1=(10,2,1)
print("s1:",s1)
s2=(10,3,1)
print("s2:",s2)
s3=(8,3,1)
print("s3:",s3)


s1s1=np.sum(np.multiply(s1,s1))
print("s1s1",s1s1)
s1s2=np.sum(np.multiply(s1,s2))
print("s1s2",s1s2)
s1s3=np.sum(np.multiply(s1,s3))
print("s1s3",s1s3)
s2s2=np.sum(np.multiply(s2,s2))
print("s2s2",s2s2)
s2s3=np.sum(np.multiply(s2,s3))
print("s2s3",s2s3)
s3s3=np.sum(np.multiply(s3,s3))
print("s3s3",s3s3)

arr=np.array(((s1s1,s1s2,s1s3),(s1s2,s2s2,s2s3),(s1s3,s2s3,s3s3)))
print("arr=",arr)
arr_side=np.array([-1,-1,1])

x = np.linalg.solve(arr, arr_side)
print("X=",x)

b=np.multiply(x[0],s1)+np.multiply(x[1],s2)+np.multiply(x[2],s3)
Final_values = list(np.around(np.array(b),2))
print("Final_values=",Final_values)

print("Weights are as followed:\nW1 =",Final_values[0],"\nW2 =",Final_values[1],"\nBias is B =",Final_values[2])

b=-Final_values[2]

plt.scatter(df['X'],df['Y'])

if(Final_values[0]==1 and Final_values[1]==0) :
    plt.plot([b, b], [-b, b+10], 'k-', lw=2)
elif(Final_values[0]==0 and Final_values[1]==1) :
    plt.plot([-b, b + 10],[b, b], 'k-;', lw=2)

plt.show()