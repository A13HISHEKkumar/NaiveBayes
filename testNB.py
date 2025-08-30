
import numpy as np
from sklearn import datasets
from NaiveBayes import NaiveBayes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

bc = datasets.load_breast_cancer()
X,y = bc.data,bc.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

clf = NaiveBayes()

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

accuracy = np.sum(y_pred == y_test) / len(y_test)
print(accuracy)

