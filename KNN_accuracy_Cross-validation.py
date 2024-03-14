# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 08:33:43 2023

@author: riadhbabaali
"""
from   sklearn.model_selection import cross_val_score
from   sklearn.model_selection import train_test_split
from   sklearn.neighbors import KNeighborsClassifier
from   sklearn.metrics import confusion_matrix
import pandas as pd

df = pd.read_csv("diabete.csv")
#print( df.head() )
y = df.iloc[:, 8]
X = df.iloc[:, :8]
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.6)

model= KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
#Holdout
print( model.score(X_train,y_train)) 
print( model.score(X_test,y_test) )

#Cross
res=cross_val_score( KNeighborsClassifier( 3 ), X,y, cv=5, scoring='accuracy').min() 
print(res)


print( confusion_matrix(y_test, model.predict(X_test) )  )
