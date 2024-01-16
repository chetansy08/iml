import numpy as np
import pandas as pd
from sklearn import svm

data=pd.read_csv("iris-data.csv")

x=data.iloc[:,:-1]
y=data.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  = train_test_split(x,y,test_size=0.2,random_state=42)

clf=svm.SVC(kernel='rbf')
clf.fit(x_train,y_train)

prediction=clf.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(prediction,y_test)
print("Accuracy:",accuracy)
