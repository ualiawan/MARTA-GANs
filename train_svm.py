# -*- coding: utf-8 -*-

from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np

accuracy = []

x_train=np.load('features/features_train.npy')
y_train=np.load('features/labels_train.npy')
x_test=np.load('features/features_test.npy')
y_test=np.load('features/labels_test.npy')

print("Fitting the classifier to the training set")
C = 1000.0  # SVM regularization parameter
clf = svm.SVC(kernel='linear', C=C).fit(x_train, y_train)

print("Predicting...")

y_pred = clf.predict(x_test)

print( "Accuracy: %.3f" %(accuracy_score(y_test, y_pred)))
accuracy.append(accuracy_score(y_test, y_pred))
print(accuracy)