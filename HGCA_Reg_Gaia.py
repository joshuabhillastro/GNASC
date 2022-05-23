#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:46:23 2022

@author: joshhill
"""
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from astroquery.gaia import Gaia
from sklearn.ensemble import RandomForestRegressor as rndfor 
import pandas as pd
import sys

dft = pd.read_csv("/Users/joshhill/Gaia Codes/hgca.csv")
df = pd.read_hdf("hipgdr2.h5")

msk = np.isin(dft.source_id,df.dr3_source_id) # find which sources are in dr2 and edr3
dfx = dft[msk].copy() # this has the cross matched sources
print('    edr3 w/dr2 srcs identified:',len(dfx.index))

# now line up the dataframes, so they are sorted by edr3 source_id (and have reset indices)
dfx.sort_values(by='source_id',inplace = True)
dfx.reset_index(drop=True,inplace=True) # drop means don't add a new index column, stackoverflow.com
df.sort_values(by='dr3_source_id',inplace=True)
df.reset_index(drop=True,inplace=True)
did = (dfx.source_id - df.dr3_source_id)
print('id cross check:that edr3 and dr2 dataframes are aligned by index',np.sum(did!=0))
#print(len(dfx.index))

#number of samples 
n_samples = len(dfx.parallax)
imshape = dfx.parallax[0].shape

#data to train on
data = np.column_stack((dfx.pmra**2+dfx.pmdec**2,dfx.pmra_error**2+dfx.pmdec_error**2,dfx.parallax, dfx.gmag, 
                            dfx.parallax_over_error, dfx.astrometric_excess_noise, dfx.astrometric_gof_al,))
nstacks = data.shape[-1]

msk = np.isfinite(data[:,0])  # get rid of infinities/nans
for i in range(1,nstacks):
        msk &= np.isfinite(data[:,i]) 
dfnew = dfx.iloc[msk]
data = data[msk]
data = (data-np.mean(data,axis=0))/np.var(data,axis=0)

#print(len(df.index))

#targ = np.log10(chi2) #how to subtract pmra and pmdec from both catalogs?
targ = np.column_stack((dfx.pmra - df.pmra,dfx.pmdec - df.pmdec,))

regressor = rndfor(n_estimators=100)

X_train, X_test, y_train, y_test = train_test_split(data[:], targ[:],
                                                    test_size=0.5,shuffle=True)
X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)

n_train = len(y_train)
n_test = len(y_test)

res = regressor.fit(X_train, y_train)

predtrain = regressor.predict(X_train)
predtest = regressor.predict(X_test)

thrval = 250
accval = 10
chi2_test = 10**y_test
chi2accpredicted = 10**predtest
groundtrupos = np.sum((chi2_test>=accval))
groundtruneg = np.sum((chi2_test<accval))
trupos = np.sum((chi2_test>=accval)&(chi2accpredicted>=thrval))
obspos = np.sum((chi2accpredicted>=thrval))
print(groundtrupos)
print(groundtruneg)
print(trupos)
print(obspos)
print(trupos/obspos)

#sys.exit()

#plot
fig = plt.figure(figsize = (10,10))
plt.subplot(2,1,1)
plt.scatter(np.log10(predtrain),np.log10(y_train), color = 'blue', s=.5, label = 'training')
plt.xlabel('predicted training')
plt.ylabel('actual')
plt.subplot(2,1,2)
plt.scatter(np.log10(predtest), np.log10(y_test), color = 'r', s=.5, label = 'testing')
plt.xlabel('predicted testing')
plt.ylabel('actual')
