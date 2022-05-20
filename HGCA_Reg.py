#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:10:31 2022

@author: joshhill
"""
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from astroquery.gaia import Gaia
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestRegressor as rndfor 
import pandas as pd

dft = pd.read_csv("/Users/joshhill/Gaia Codes/hgca.csv")

#Building in parameters


#call data lines of intrest
a_gof = dft.astrometric_gof_al.to_numpy()
a_en = dft.astrometric_excess_noise.to_numpy()
pmra = dft.pmra.to_numpy()
pmdec = dft.pmdec.to_numpy()
pmra_e = dft.pmra_error.to_numpy()
pmdec_e = dft.pmdec_error.to_numpy()
par = dft.parallax.to_numpy()
gmag = dft.gmag.to_numpy()
chi2 = dft.chi2acc.to_numpy()
paroe = dft.parallax_over_error.to_numpy()

#number of samples 
n_samples = len(par)
imshape = par[0].shape

#data to train on
nstacks = 7
data = np.column_stack((pmra**2+pmdec**2,pmra_e**2+pmdec_e**2,par, gmag, 
                            paroe, a_en, a_gof,))

msk = np.isfinite(data[:,0])  # get rid of infinities/nans
for i in range(1,nstacks):
        msk &= np.isfinite(data[:,i]) 
dfnew = dft.iloc[msk]
data = data[msk]
data = (data-np.mean(data,axis=0))/np.var(data,axis=0)

targ = np.log10(chi2) #I have no idea what to put here
#targ = data 

regressor = rndfor(n_estimators=100)

X_train, X_test, y_train, y_test = train_test_split(data[:], targ[:],
                                                    test_size=0.5,shuffle=True)
n_train = len(y_train)
n_test = len(y_test)

res = regressor.fit(X_train, y_train)

predtrain = regressor.predict(X_train)
predtest = regressor.predict(X_test)


#plot
fig = plt.figure(figsize = (10,10))
plt.subplot(2,1,1)
plt.scatter(np.log10(predtrain),np.log10(y_train), s =.5, color = 'blue', label = 'training')
plt.xlabel('predicted training')
plt.ylabel('actual training')
plt.subplot(2,1,2)
plt.scatter(np.log10(predtest), np.log10(y_test), s = .5, color = 'r', label = 'testing')
plt.xlabel('predicted testing')
plt.ylabel('actual testing')

