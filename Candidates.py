#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:33:48 2022

@author: joshhill
"""
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from astropy.table import Table
from astropy.io.votable import from_table, writeto
import numpy as np
from astroquery.gaia import Gaia
from sklearn.ensemble import RandomForestRegressor as rndfor 
import pandas as pd
import sys
import pickle
from csv import reader

#dfa = pd.read_csv("/Users/joshhill/Gaia Codes/hgca.csv")
df2 = pd.read_csv("/Users/joshhill/hipgdr2.csv")
df3 = pd.read_csv("/Users/joshhill/hipgedr3.csv")
dfrnd2 = pd.read_csv("/Users/joshhill/dr2rnd.csv")
dfrnd3 = pd.read_csv("/Users/joshhill/edr3rnd.csv")



#add a def to correct for the frame rotation
pmra_cat = dfrnd2['pmra']
pmdec_cat = dfrnd2['pmdec']

w_x = -.077
w_y = -.096
w_z = -.002


def frameadjust(pmra,pmdec):
    deg2rad = np.pi/180
    a = w_x*np.cos(pmra*deg2rad)*np.sin(pmdec*deg2rad)
    b = w_y*np.sin(pmra*deg2rad)*np.sin(pmdec*deg2rad)
    c = w_z*np.cos(pmdec*deg2rad)
    
    pmra_true = a+b-c
        
    d = w_x*np.sin(pmdec*deg2rad)
    e = w_y*np.cos(pmdec*deg2rad)
    
    
    pmdec_true = e-d
    
    return (pmra_true, pmdec_true)


pmra, pmdec = frameadjust(pmra_cat,pmdec_cat)
#think about a name for the catalog
#<100 pc or parallax >0.01 from edr3 xmatch w/ dr2 get rid of anything wierd 
#if mags dont match within a half a mag get rid look at drb example
#how to check if we are right?
#methods for research? maybe our project as an example?

#print(dfa.columns)

#sys.exit()

np.random.seed(777134134)

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


msk = np.isin(dfrnd3.source_id,dfrnd2.dr3_source_id) # find which sources are in dr2 and edr3
dfrndx = dfrnd3[msk].copy() # this has the cross matched sources
print('    edr3rnd w/dr2rnd srcs identified:',len(dfrndx.index))

# now line up the dataframes, so they are sorted by edr3 source_id (and have reset indices)
dfrndx.sort_values(by='source_id',inplace = True)
dfrndx.reset_index(drop=True,inplace=True) # drop means don't add a new index column, stackoverflow.com
dfrnd2.sort_values(by='dr3_source_id',inplace=True)
dfrnd2.reset_index(drop=True,inplace=True)
did1 = (dfrndx.source_id - dfrnd2.dr3_source_id)
print('dr2rnd/edr3rnd id cross check, 0 is good:',np.sum(did1!=0)) 
#sys.exit()


#number of samples 
n_samples = len(dfrndx.parallax)
imshape = dfrndx.parallax[0].shape
pmrnd = np.sqrt(dfrndx.pmra**2+dfrndx.pmdec**2) 
pmrnd3 = np.sqrt(dfrnd3.pmra**2+dfrnd3.pmdec**2) 
pm = np.sqrt(dfx.pmra**2+dfx.pmdec**2)
dfx = dfx.sample(frac=1)
print("souce id 1:", dfx['source_id'].iloc[0])
data = np.column_stack((np.sqrt(dfx.pmra**2+dfx.pmdec**2),
                        np.sqrt(dfx.pmra_error**2+dfx.pmdec_error**2),
                        np.sqrt((df2.pmdec-dfx.pmdec)**2+(df2.pmra-dfx.pmra)**2),
                        dfx.pmra, dfx.pmdec,dfx.pmra_error,
                        dfx.pmdec_error,df2.pmra-dfx.pmra,df2.pmdec-dfx.pmdec, 
                        dfx.ruwe,dfx.parallax, dfx.parallax_over_error,dfx.phot_g_mean_mag,
                        dfx.bp_rp,dfx.astrometric_excess_noise_sig,dfx.astrometric_gof_al),)
#dfrndx = dfrndx.sample(frac=1)
datarnd = np.column_stack((np.sqrt(dfrndx.pmra**2+dfrndx.pmdec**2),
                           np.sqrt(dfrndx.pmra_error**2+dfrndx.pmdec_error**2),
                           np.sqrt((dfrnd2.pmdec-dfrndx.pmdec)**2+(dfrnd2.pmra-dfrndx.pmra)**2),
                           dfrndx.pmra, dfrndx.pmdec,dfrndx.pmra_error, dfrndx.pmdec_error,
                           dfrnd2.pmra-dfrndx.pmra,dfrnd2.pmdec-dfrndx.pmdec, dfrndx.ruwe,
                           dfrndx.parallax, dfrndx.parallax_over_error,dfrndx.phot_g_mean_mag,
                           dfrndx.bp_rp,dfrndx.astrometric_excess_noise_sig,
                           dfrndx.astrometric_gof_al))# feed into predictor
nstacks = datarnd.shape[-1]

msk = np.isfinite(datarnd[:,0])  # get rid of infinities/nans
for i in range(1,nstacks):
        msk &= np.isfinite(datarnd[:,i]) 
dfrndx = dfrndx.iloc[msk].copy()
datarnd = datarnd[msk]
#datarnd = (datarnd-np.mean(datarnd,axis=0))/np.std(datarnd,axis=0)

nstacks = data.shape[-1]

msk = np.isfinite(data[:,0])  # get rid of infinities/nans
for i in range(1,nstacks):
        msk &= np.isfinite(data[:,i]) 
dfx = dfx.iloc[msk].copy()
data = data[msk]
#data = (data-np.mean(data,axis=0))/np.std(data,axis=0)

dfrndx = dfrndx[dfrndx.phot_g_mean_mag < 17.5].copy()
dfrndx.sort_values(by='source_id',inplace = True)
dfrndx.reset_index(drop=True,inplace=True)

msk = np.isin(dfrndx.source_id,dfx.source_id)
dfrndx = dfrndx[~msk].copy()
dfrndx.sort_values(by='source_id',inplace = True)
dfrndx.reset_index(drop=True,inplace=True)

targ = np.log10(dfx.chi2acc) #how to subtract pmra and pmdec from both catalogs?
#pm does not seem to make a difference

regressor = rndfor(n_estimators=150)

#X_train, X_test, y_train, y_test = train_test_split(data[:], targ[:],
#                                                   test_size=0.5,shuffle=True)
ndata = len(dfx.index)
values = dfx.chi2acc
ntrain = 4*ndata//8; ntest = ndata-ntrain
print('train set: ', ntrain)
        
Xtrain = data[:ntrain]; ytrain = np.log10(values[:ntrain])
Xtest = data[ntrain:];  ytest = values[ntrain:]
print('first element in ytrain:', ytrain[0])
print('last element in Xtrain:', Xtrain[-1,-1])
#sys.exit()
#X_train = data[:]
#y_train = targ[:]

#X_rndtrain = np.column_stack((dfrndx.pmra-dfrnd2.pmra,dfrndx.pmdec-dfrnd2.pmdec,dfrndx.astrometric_excess_noise_sig,dfrndx.astrometric_gof_al,
#                            dfrndx.parallax,dfrndx.parallax_over_error,dfrndx.pmra,dfrndx.pmdec,pmrnd3))

n_train = len(ytrain)
#n_test = len(y_test)

res = regressor.fit(Xtrain, ytrain)

predtrain = regressor.predict(Xtrain) #waas this looking at linear drift and not acc?
print('predicted',predtrain[0])
#sys.exit()
#predtest = regressor.predict(X_test)
predrnd = regressor.predict(datarnd) #random catalog



#sys.exit()
#do a np.where??
msk = np.where(10**predrnd > 28.75) #chi2acc look at different vals 
dfrndacc = dfrnd3.iloc[msk].copy()
srclis = dfrndacc['source_id'].to_list()
chi2acc = 10**predrnd[msk]
#2800 number of stars above 28.75 
err = np.sqrt(len(srclis))
print("pisson err", err)

#chi2acc = 10**4 number of stars = 0
#chi2acc = 5000 number of stars = 0
#chi2acc = 1000 number of stars = 15
#chi2acc = 500 number of stars = 52
#chi2acc = 400 number of stars = 65
#chi2acc = 300 number of stars = 97
#chi2acc = 200 number of stars = 208
#chi2acc = 100 number of stars = 1401
#chi2acc = 50 number of stars = 4856
#chi2acc = 28.75 number of stars = 14616
#chi2acc = 11.8 number of stars = 56339 fluctuates around 56000
print('numbers of stars: ', len(srclis))
#sys.exit()
#print(srclis)
#xmach with known accelerating stars
#think about training nss catalog then test on our accelerating catalog?
#top acc look at the fifty stars in simbad!! aladin lite look of anonomolis things and streaks in the stars
#dss2/blue do this on chrome
#make sure to query starting with gaia edr3, or using coordinates (ra, dec)
#investigate see if there is anything wired!!
#look and see if there is anything unusual about the surrounding area (like binaries)
#if high proper motion use wiseview nasa byw.tools 
#build table in latex and in a readable format for the journal in python 
#limit b and l in galactic coordinates
#look for an orbital period of around 1 year
#look at stars first then use the data that ben gives you depending on what he gives
#sys.exit()

#for i, row in dfrndacc.iterrows():
    #print(row['source_id'], 10**predrnd[i])
#    chi2acc = (row['source_id'], 10**predrnd[i])

srcchi = [srclis, chi2acc]
dftable = pd.DataFrame(srcchi).transpose()
dftable.columns = ["source_id", "chi2accpredicted"]
dftable.to_csv('/Users/joshhill/ourcatalog3.csv')
#sort by chi2
dftable.sort_values(by = 'chi2accpredicted', ascending = False, ignore_index = True, inplace = True)
final = dftable.head(25)
print(final)


#data5 = ([dfrndacc.source_id], [10**predrnd[i]])

#get to csv from pandas 
sys.exit()


    