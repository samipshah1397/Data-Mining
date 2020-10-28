# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 02:28:12 2020

@author: Samip
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import squarify
import reverse_geocoder as rg 
import pprint 
from pygeocoder import Geocoder
import geopandas

df = pd.read_csv("earthquake.csv")
print(df)
print(df.shape)
print(df.columns.values)

print("Nan in each columns" , df.isnull().sum(), sep='\n')

net = df.groupby('net').net.count()
print("net",net)

df['mag'].fillna(df['mag'].mean(), inplace=True)
print("Nan in each columns" , df.isnull().sum(), sep='\n')

magnitude_type = df.groupby('magType').magType.count()
print("magnitude_type",magnitude_type)

df['magType'] = df['magType'].replace(np.nan, 'ml')
print("Nan in each columns" , df.isnull().sum(), sep='\n')

earthquake_type = df.groupby('type').magType.count()
print("type",earthquake_type)


magnitude_type = df.groupby('type')['mag'].mean()
print("magnitude_type",magnitude_type)

depth_type = df.groupby('type')['depth'].mean()
print("depth_type",depth_type)

# fig, (axis1,axis2) = plt.subplots(1, 2, figsize=(7,5))


# ax = magnitude_type.plot.bar(ax=axis1, title='magnitude_type', sharey=True)
# # ax.set_ylim(-2.0,2.0)
# ax = depth_type.plot.bar(ax=axis2, title='depth_type', sharey=True)
# # ax.set_ylim(-2.0,20.0)


magnitude1_type = df.groupby('magType')['mag'].mean()
print("magnitude1_type",magnitude1_type)

depth1_type = df.groupby('magType')['depth'].mean()
print("depth1_type",depth1_type)

fig, (axis1,axis2) = plt.subplots(1, 2, figsize=(7,5))


ax = magnitude1_type.plot.bar(ax=axis1, title='magnitude1_type', sharey=True)
# ax.set_ylim(-2.0,2.0)
ax = depth1_type.plot.bar(ax=axis2, title='depth1_type', sharey=True)
# ax.set_ylim(-2.0,20.0)



# from mpl_toolkits.basemap import Basemap

# m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')

# longitudes = df["longitude"].tolist()
# latitudes = df["latitude"].tolist()
# #m = Basemap(width=12000000,height=9000000,projection='lcc',
#             #resolution=None,lat_1=80.,lat_2=55,lat_0=80,lon_0=-107.)
# x,y = m(longitudes,latitudes)

# fig = plt.figure(figsize=(12,10))
# plt.title("All affected areas")
# m.plot(x, y, "o", markersize = 2, color = 'blue')
# m.drawcoastlines()
# m.fillcontinents(color='coral',lake_color='aqua')
# m.drawmapboundary()
# m.drawcountries()
# plt.show()