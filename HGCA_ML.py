#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:16:07 2022

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

#DR2 source ID
query = ("select top 31123 "
                      "solution_id,astrometric_n_obs_al,astrometric_gof_al,astrometric_primary_flag,astrometric_excess_noise,phot_g_mean_mag,parallax_error,parallax_over_error,bp_rp,phot_g_mean_mag,parallax, "
                      "duplicated_source "
                      "from gaiaedr3.gaia_source order by source_id")
job = Gaia.launch_job(query=query)
r = job.get_results()

#X train will be things like (parallax, error, color, the various flags, A-Goff, etc)
astrometic_obs = r['astrometric_n_obs_al'].data #Note that some observations may be strongly downweighted (see astrometric_n_bad_obs_al)
a_gof = r['astrometric_gof_al'].data #Goodness of fit statistic of model wrt along-scan observations
astrometic_flag = r['astrometric_primary_flag'].data ##Flag indicating if this source was used as a primary source (true) or secondary source (false)
astrometic_noise = r['astrometric_excess_noise'].data #This is the excess noise 𝜖𝑖 of the source. It measures the disagreement, expressed as an angle, between the observations of a source and the best-fitting standard astrometric model 
g_mean_mag = r['phot_g_mean_mag'].data #Mean magnitude in the G band.
parallax_error = r['parallax_error'].data
parallax_over_error = r['parallax_over_error'].data


#Creating M_G to use with ML
bp_rp = r['bp_rp'].data
M_g = r['phot_g_mean_mag'].data
Par = r['parallax'].data
d = 1/Par
D = d*1000
M_G = M_g - 5*np.log10(D/10)


#For now I will use parallax for what the practice lab uses as digits
n_samples = len(Par) 
imshape = Par[0].shape #not sure about this line

#data = Par.reshape((n_samples, 10000))
n_parms = 3
data = np.zeros((n_samples,n_parms))
data [:,0]= np.nan_to_num(Par)
data [:,1]= np.nan_to_num(M_g)
data [:,2]= np.nan_to_num(bp_rp)

table = pd.read_csv("/Users/joshhill/Gaia Data/HGCA_Accel.csv", usecols = [1,2,3])
bp_rpt = table.iloc[:,1]
M_gt = table.iloc[:,2]
part = table.iloc[:,2]
dt = 1/part
Dt = dt*1000
M_Gt = M_gt -5*np.log10(Dt/10)
print(M_Gt,bp_rpt)

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