
from   sklearn.model_selection import validation_curve
from   sklearn.model_selection import train_test_split
from   sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("diabete.csv")
#print( df.head() )
y = df.iloc[:, 8]
X = df.iloc[:, :8]
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.5)
#print(X_train.shape)

model= KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

k=np.arange(1,20)
train_score, val_score = validation_curve( model, X_train, y_train, 'n_neighbors', k, cv=5)
test_score, val_score2 = validation_curve( model, X_test, y_test, 'n_neighbors', k, cv=5)

plt.plot( k, val_score.mean(axis=1),c='r')
plt.plot( k, val_score2.mean(axis=1))
plt.show()

