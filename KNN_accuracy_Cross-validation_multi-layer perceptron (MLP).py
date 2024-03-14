
from   sklearn.model_selection import train_test_split
from   sklearn.neural_network import MLPClassifier
import pandas as pd

df = pd.read_csv("diabete.csv")
#print( df.head() )
y = df.iloc[:, 8]
X = df.iloc[:, :8]
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.6)

model= MLPClassifier(max_iter=300, activation="relu", hidden_layer_sizes=(100,100) )
model.fit(X_train,y_train)

print( model.score(X_train,y_train)) 
print( model.score(X_test,y_test) )

