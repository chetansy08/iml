import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("breast-cancer.csv")

data["diagnosis"].value_counts()

from sklearn import preprocessing
label_encode =preprocessing.LabelEncoder()
labels = label_encode.fit_transform(data["diagnosis"])
data["target"] =labels
data.drop(columns="diagnosis", axis=1, inplace=True)

X=data.iloc[:,:-1]
y=data.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.25,random_state=0)

from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier()
classifier.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
