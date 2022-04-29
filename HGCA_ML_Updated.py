#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:47:42 2022

@author: marcwhiting
"""

import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from astroquery.gaia import Gaia
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import colors
import pandas as pd
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

#Read in the HGCA table to begin ML training
table = pd.read_csv("/Users/marcwhiting/Desktop/Anaconda/Planet 9/HGCA_Accel.csv", usecols = [1,2,3])
bp_rpt = table.iloc[:,0]
M_gt = table.iloc[:,1]
part = table.iloc[:,2]

#To get M_Gt, also using the mskt to remove the error given 
dt = 1/part
Dt = dt*1000
mskt = (Dt>0)&(Dt<100000) 
bp_rpt = bp_rpt[mskt]
M_gt = M_gt[mskt]
part = part[mskt]
dt = dt[mskt]
Dt = Dt[mskt] 
M_Gt = M_gt -5*np.log10(Dt/10)


n_samples = len(part) 
imshape = part[0].shape #not sure what this line does


n_parms = 2
data = np.zeros((n_samples,n_parms))
#data [:,0]= np.nan_to_num(part)
data [:,0]= np.nan_to_num(M_gt)
data [:,1]= np.nan_to_num(bp_rpt)

targ = np.zeros(n_samples)
targ = np.where((M_Gt>0.0)&(M_Gt<1.0)&(bp_rpt>1.1)&(bp_rpt<1.3),1,0)



#classifier = svm.SVC(kernel="poly", degree=5)
classifier = svm.SVC(gamma = 5)


#classifier = BaggingClassifier(KNeighborsClassifier())
#I KEPT Bagging in for now, Accuracy is comparable, perhaps find a way to adjust


# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data[:], targ[:],
                                                    test_size=0.5,shuffle=True)
n_train = len(y_train)
n_test = len(y_test)

#30704
#31123 
query = ("select top 30704 "
                      "bp_rp,phot_g_mean_mag,parallax, "
                      "duplicated_source "
                      "from gaiaedr3.gaia_source where parallax_over_error >= 10 AND b > 20 order by random_index")
job = Gaia.launch_job(query=query)
r = job.get_results()

#Creating M_G to use with ML
bp_rp = r['bp_rp'].data
M_g = r['phot_g_mean_mag'].data
Par = r['parallax'].data
d = 1/Par
D = d*1000 
M_G = M_g - 5*np.log10(D/10)



data1 = np.zeros((n_samples,n_parms))
#data1 [:,0]= np.nan_to_num(Par)
data1 [:,0]= np.nan_to_num(M_G)
data1 [:,1]= np.nan_to_num(bp_rp)


X_train1, X_test1, y_train1, y_test1 = train_test_split(data[:], targ[:],
                                                    test_size=0.5,shuffle=True)
classifier.fit(X_train, y_train)
#15561
#15352
test = classifier.predict(data1[15352:,:])
print(np.sum(targ==1))

#To see what we got right more easily
print('number in test set:',len(y_test1))
right=np.sum(y_test1==test)
print('number correctly classified:',right)


msk = y_test!=test
X_fails,y_fails,y_failspred = X_test1[msk],y_test1[msk],test[msk]
X, y = make_classification(random_state=0)

cm = confusion_matrix(y_test1, test, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classifier.classes_)
disp.plot()
plt.show()



#now create a CMD
h = plt.hist2d(bp_rp, M_G, bins=300, cmin = 2,  range = [[-4,8],[-3,20]],
                                   norm=colors.PowerNorm(0.5), zorder=2.5)
plt.scatter(bp_rp,M_G,s=.5, color='k', zorder=0)
plt.ylim(20,-3)
plt.xlim(-4,8)
plt.xlabel('bp-rp')
plt.ylabel('M_G')
cb = plt.colorbar(h[3], ax=plt.subplot(), pad=0.02)
cb.set_label('Stellar Density')