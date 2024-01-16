import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Loading data
irisData = load_iris()

# Create feature and target arrays
X = irisData.data
y = irisData.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)

# Predict on dataset which model has not seen before
y_pred=knn.predict(X_test)

score = knn.score(X_test, y_test)
print(y_pred)
print(f"score : {score:.3f}")

cor_cls, incor_cls = 0, 0

for actual, pred in zip(y_pred, y_test):
    if actual==pred:
#         print(f'Correct classification : {actual} and predicted : {pred}')
        cor_cls+=1
    else :
#         print(f'Incorrect classification : {actual} and predicted : {pred}')
        incor_cls+=1

print(f"No. of correct classifications : {cor_cls}")
print(f"No. of correct classifications : {incor_cls}")






""" extra """


# import matplotlib.pyplot as plt
# neighbors = np.arange(1, 9)
# train_accuracy = np.empty(len(neighbors))
# test_accuracy = np.empty(len(neighbors))

# # Loop over K values
# for i, k in enumerate(neighbors):
# 	knn = KNeighborsClassifier(n_neighbors=k)
# 	knn.fit(X_train, y_train)

# 	# Compute training and test data accuracy
# 	train_accuracy[i] = knn.score(X_train, y_train)
# 	test_accuracy[i] = knn.score(X_test, y_test)

# # Generate plot
# plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
# plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')

# plt.legend()
# plt.xlabel('n_neighbors')
# plt.ylabel('Accuracy')
# plt.show()
