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

dfa = pd.read_csv("/Users/joshhill/Gaia Codes/hgca.csv")
df2 = pd.read_hdf("/Users/joshhill/hipgdr2.h5")
df3 = pd.read_hdf("/Users/joshhill/hipgedr3.h5")

#print(dfa.columns)
print(np.sum(df3.source_id-dfa.source_id),'checking hgca and edr3 0 is good!')
df3['chi2acc'] = dfa['chi2acc']

#sys.exit()

msk = np.isin(df3.source_id,df2.dr3_source_id) # find which sources are in dr2 and edr3
dfx = df3[msk].copy() # this has the cross matched sources
print('    edr3 w/dr2 srcs identified:',len(dfx.index))

# now line up the dataframes, so they are sorted by edr3 source_id (and have reset indices)
dfx.sort_values(by='source_id',inplace = True)
dfx.reset_index(drop=True,inplace=True) # drop means don't add a new index column, stackoverflow.com
df2.sort_values(by='dr3_source_id',inplace=True)
df2.reset_index(drop=True,inplace=True)
did = (dfx.source_id - df2.dr3_source_id)
print('dr2/edr3 id cross check, 0 is good:',np.sum(did!=0)) 
#print(len(dfx.index)

srcid = np.array(df3.source_id)                                                                                           
print('edr3 cf.',len(srcid),len(np.unique(srcid)))                                                                        
#check = np.sum(df3.source_id != elis)                                                                                     
#print('edr3 v. hgca source id check. 0 is good:',check)                                                                   
#df3['chi2acc']=chi2lis          # save chi2acc here! could use this the ML target/label.                                                                   
                                                                       
check = np.sum(df3.source_id.values != dfx.source_id.values) + np.sum(df2.source_id.values != dfx.chi2acc.values)                                      
print(' checking hgca, new edr3, 0 is good:',check)                                                                       
# going forward I prolly won't use chi2lis, but dfx.chi2acc instead...

#number of samples 
n_samples = len(dfx.parallax)
imshape = dfx.parallax[0].shape

#data to train on
#data = np.column_stack((dfx.pmra**2+dfx.pmdec**2,dfx.pmra_error**2+dfx.pmdec_error**2,dfx.parallax, dfx.gmag, 
                            #dfx.parallax_over_error, dfx.astrometric_excess_noise, dfx.astrometric_gof_al,))
data = np.column_stack((dfx.pmra-df2.pmra,dfx.pmdec-df2.pmdec,))
nstacks = data.shape[-1]

#msk = np.isfinite(data[:,0])  # get rid of infinities/nans
#for i in range(1,nstacks):
#        msk &= np.isfinite(data[:,i]) 
#dfnew = dfx.iloc[msk]
#data = data[msk]
#data = (data-np.mean(data,axis=0))/np.var(data,axis=0)
#doing this above gives a smaller len(data) value how to fix?

#print(len(df.index))

#pmx = np.sqrt((dfx.pmra)**2 + (dfx.pmdec)**2)
#pm = np.sqrt(df.pmra**2+df.pmdec**2) 
pm = np.sqrt((dfx.pmra - df2.pmra)**2 + (dfx.pmdec - df2.pmdec)**2)

targ = np.log10(dfx.chi2acc) #how to subtract pmra and pmdec from both catalogs?
#targ = np.column_stack((dfx.pmra - df2.pmra,dfx.pmdec - df2.pmdec,pm,)) #Does pm matter or not?
#pm does not seem to make a difference

regressor = rndfor(n_estimators=100)

X_train, X_test, y_train, y_test = train_test_split(data[:], targ[:],
                                                    test_size=0.5,shuffle=True)
X_train = np.nan_to_num(X_train)
y_train = np.nan_to_num(y_train)
X_test = np.nan_to_num(X_test)
y_test = np.nan_to_num(y_test)

n_train = len(y_train)
n_test = len(y_test)

res = regressor.fit(X_train, y_train)

predtrain = regressor.predict(X_train)
predtest = regressor.predict(X_test)

thrval = 250
accval = 10 #chi2
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
plt.scatter(predtrain, y_train, color = 'blue', s=.5, label = 'training')
plt.xlabel('predicted training')
plt.ylabel('actual')
plt.subplot(2,1,2)
plt.scatter(predtest, y_test, color = 'r', s=.5, label = 'testing')
plt.xlabel('predicted testing')
plt.ylabel('actual')
