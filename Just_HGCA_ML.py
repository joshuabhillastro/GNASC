#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:55:33 2022

@author: joshhill
"""
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from astroquery.gaia import Gaia
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
import pandas as pd
#from astropy.table import Table
from sklearn import datasets, svm, metrics
#import matplotlib as mpl

table = pd.read_csv("/Users/joshhill/Gaia Data/HGCA_Accel.csv", usecols = [1,2,3])
bp_rp = table.iloc[:,0]
M_g = table.iloc[:,1]
Par = table.iloc[:,2]
d = 1/Par
D = d*1000
M_G = M_g -5*np.log10(D/10)


#For now I will use parallax for what the practice lab uses as digits
n_samples = len(Par) 
imshape = Par[0].shape #not sure about this line

#data = Par.reshape((n_samples, 10000))
n_parms = 3
data = np.zeros((n_samples,n_parms))
data [:,0]= np.nan_to_num(Par)
data [:,1]= np.nan_to_num(M_g)
data [:,2]= np.nan_to_num(bp_rp)

targ = np.zeros(n_samples)
targ = np.where((M_G<3)&(M_G>-2)&(bp_rp>1.0)&(bp_rp<1.8),1,0)
print(np.sum(targ))

# choose your classifier. I stuck with the one from the lab, but this is tbd
classifier = svm.SVC()

print(data.shape)
# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data[:], targ[:],
                                                    test_size=0.5,shuffle=True)
n_train = len(y_train)
n_test = len(y_test)

#X_train = X_train.reshape(1,-1)
#y_train = y_train.reshape(1,-1)

print(X_train.shape,y_train.shape)
classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)

print('number in test set:',len(y_test))
right=np.sum(y_test==predicted)
print('number correctly classified:',right)

msk = y_test!=predicted
X_fails,y_fails,y_failspred = X_test[msk],y_test[msk],predicted[msk]
X, y = make_classification(random_state=0)
#X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                  #  random_state=0)
#clf = SVC(random_state=0)
#clf.fit(X_train, y_train)
#SVC(random_state=0)
#predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, predicted, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classifier.classes_)
disp.plot()
plt.show()