import numpy as np
import pandas as pd

dataset=pd.read_csv("/home/dextro/Desktop/ML_FINAL/Codes/KNN/sample.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,2].values

#for plotting
x1=dataset.iloc[:,:-2].values
y1=dataset.iloc[:,-2].values
import matplotlib.pyplot as plt
plt.scatter(x1,y1,color="red")
plt.show()

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=3)
classifier.fit(X,y)
H=int(input("Enter Height:"))
W=int(input("Enter Weight:"))
y_predict=classifier.predict([[H,W]])

print('General KNN:' , y_predict)

classifier=KNeighborsClassifier(n_neighbors=3, weights='distance')
classifier.fit(X,y)
y_pred=classifier.predict([[H,W]])
print('Distance weighted KNN:' , y_pred)

