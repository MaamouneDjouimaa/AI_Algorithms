# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 23:02:47 2023
@author: riadhbabaali
"""
from   sklearn.neighbors import KNeighborsClassifier
import pandas as pd

df = pd.read_csv("diabete.csv")
print( df.head() )

y = df.iloc[:, 8]
X = df.iloc[:, :8]

model= KNeighborsClassifier()
model.fit( X, y )

print( model.score(X,y) )
nouveau=[ [5,140,70,32,0,31.5,0.5,52], 
          [6,148,72,35,0,33.6,0.627,50] ]
#yp=model.predict(X)
print( model.predict( nouveau )  )