
from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv("meteo2.csv")
print( df.head() )

y = df.iloc[:, 2]
X = df.iloc[:, :2]

model=LinearRegression()
model.fit(X,y)
print( model.score(X,y) )

Yp=model.predict( [[990,0.89]] )
print( Yp )

