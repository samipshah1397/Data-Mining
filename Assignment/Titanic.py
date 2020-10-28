# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:54:13 2020

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

df = pd.read_csv("Titanic.csv")
print(df)
print(df.shape)
print(df.columns.values)

#print("Check NaN in Dataframe" , df.isnull(), sep='\n')
print("Nan in each columns" , df.isnull().sum(), sep='\n')

df["embarked"].replace({"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}, inplace=True)

print(df["embarked"])

Count_Embarked = df.groupby('embarked').embarked.count()
print(Count_Embarked)


df['embarked'] = df['embarked'].replace(np.nan, 'Queenstown')
print("Nan in each columns" , df.isnull().sum(), sep='\n')

df['fare'].fillna(df['fare'].mean(), inplace=True)
print("Nan in each columns" , df.isnull().sum(), sep='\n')


df['home.dest'] = df['home.dest'].replace(np.nan, 'Unknown,')
print("Nan in each columns" , df.isnull().sum(), sep='\n')


df.rename(columns = {'home.dest':'location'}, inplace = True) 

print(df.columns.values)

df['Home_Location'] = df['location'].str.split(',').str[0]
df = df.drop(columns=['location'])

print(df)
Count_Home_Location = df.groupby('Home_Location').Home_Location.count()
print("Count_Home_Location",Count_Home_Location)


print("Nan in each columns" , df.isnull().sum(), sep='\n')

df['age'] = df['age'].replace(np.nan, 0)

print("Nan in each columns" , df.isnull().sum(), sep='\n')

print(df.head())

#add a new column category next to the age group. 
category = pd.cut(df.age,bins=[0,1,20,60,99],labels=['Unknown','Child','Adult','Elderly'])
df.insert(5,'Age_Group',category)
print(df["Age_Group"])
 
# hello = df.groupby(['sex'])['survived'].sum()
# print(hello)
Count_sex = df['sex'].value_counts()
print(Count_sex)

# df['sex'].value_counts().plot(kind='bar');

# plt.xlabel("Gender", labelpad=14)
# plt.ylabel("Count of People", labelpad=14)
# plt.title("Count of People Who Received Tips by Gender", y=1.02);
# plt.show()

survived_by_age = df.groupby('Age_Group')['survived'].mean()
survived_by_class = df.groupby('pclass')['survived'].mean()
survived_by_sex = df.groupby('sex')['survived'].mean()
print("Survived by Class",survived_by_age)
print("Survived by Class",survived_by_class)
print("Survived by Gender",survived_by_sex)
# fig, (axis1,axis2,axis3) = plt.subplots(1, 3, figsize=(10,6))


# ax = survived_by_age.plot.bar(ax=axis1, title='Survival Rate by Age', sharey=True)
# ax.set_ylim(0.0,1.0)
# ax = survived_by_sex.plot.bar(ax=axis2, title='Survival Rate by Gender', sharey=True)
# ax.set_ylim(0.0,1.0)
# ax = survived_by_class.plot.bar(ax=axis3, title='Survival Rate by Class', sharey=True)
# ax.set_ylim(0.0,1.0)

fare_by_class_sex = df.groupby(['pclass', 'sex'])['fare'].mean()
print("Fare by Class and Gender\n",fare_by_class_sex)

survived_by_home = df.groupby('Home_Location')['survived'].count()
print(survived_by_home)

# fig, ax = plt.subplots(figsize=(10,6))
# # create data
# x = df['age']
# y = df['fare']
# z = df['pclass']

# colors = df['pclass'] 
# # use the scatter function
# plt.xlabel('age')
# plt.ylabel('fare')
# scatter = ax.scatter(x, y, s=1+(z+z*z*z),c=colors)
# legend1 = ax.legend(*scatter.legend_elements(),
#                     loc="best", title="pclass")
# plt.show()


df1 = df[['embarked','sex','survived']]
print(df1)

df1 = df1.groupby(['embarked','sex']).count().reset_index()
print(df1)

# fig, ax = plt.subplots(figsize=(12, 8))

# # Our x-axis. We basically just want a list
# # of numbers from zero with a value for each
# # of our jobs.
# x = np.arange(len(df1.sex.unique()))
# bar_width = 0.2

# b1 = ax.bar(x, df1.loc[df1['embarked'] == 'Cherbourg', 'survived'],
#             width=bar_width,label='Cherbourg')
# b2 = ax.bar(x + bar_width, df1.loc[df1['embarked'] == 'Queenstown', 'survived'],
#             width=bar_width,label='Queenstown')
# b3 = ax.bar(x + bar_width+ bar_width, df1.loc[df1['embarked'] == 'Southampton', 'survived'],
#             width=bar_width,label='Southampton')


# ax.set_xticks(x + bar_width)
# ax.set_xticklabels(df1.sex.unique())

# ax.legend()

# ax.set_xlabel('Gender', labelpad=15)
# ax.set_ylabel('Survived', labelpad=15)
# ax.set_title('Survival Count based on Embarked', pad=15)


print(df)

df = df.Home_Location.unique()
print(df)



