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

dfa = pd.read_csv("/Users/joshhill/Gaia Codes/hgca.csv")
#df2 = pd.read_hdf("/Users/joshhill/hipgdr2.h5")
#df3 = pd.read_hdf("/Users/joshhill/hipgedr3.h5")

#how to implement query?
#kenyon bromely on ads look up
#talk about who writes what
#think about a name for the catalog
#Add the cmd's to the overleaf and add units and crop and export to pdf in code

#how to query? How to combine both dr2 and edr3 
#<100pc or parallax>10 from edr3 xmatch w/ dr2 get rid of anything wierd 
#if mags dont match within a half a mag get rid look at drb example


query = ("select top 115346 "
                  "source_id, astrometric_gof_al, astrometric_excess_noise, pmra, pmdec, pmra_error, pmdec_error,"
                  "parallax, phot_g_mean_mag, parallax_over_error, astrometric_chi2_al "
                  "from gaiaedr3.gaia_source where parallax > 10 order by source_id")
job = Gaia.launch_job(query=query)
r = job.get_results()
df2 = r.to_pandas()
mytable = Table(r)
fmytable =' my_table.xml'
votable = from_table(mytable)
writeto(votable, fmytable)
upload_resource = fmytable
print("done with first query")

query = ("SELECT top 115346 gaiaedr3.*, dr2toedr3.* "
"FROM tap_upload.table_test AS gaiaedr3 "
"JOIN gaiaedr3.dr2_neighbourhood AS dr2toedr3 "
    "ON dr2toedr3.dr3_source_id = gaiaedr3.source_id "
    "where parallax > 10 and magnitude_difference < 0.5 "
"ORDER BY gaiaedr3.source_id ASC ")
job = Gaia.launch_job(query=query, upload_resource=upload_resource, upload_table_name="table_test", verbose=True)
r2 = job.get_results()
df3 = r2.to_pandas()

print('done with second query')

#sys.exit()

#print(dfa.columns)
print(np.sum(df3.source_id-dfa.dr3_source_id),'checking hgca and edr3 0 is good!')
df3['chi2acc'] = dfa['chi2acc'] #get acc in place

#sys.exit()

msk = np.isin(df3.source_id,df2.source_id) # find which sources are in dr2 and edr3
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
#dfx['dr2_pmra'] = df2.pmra
#dfx['dr2_pmra_e'] = df2.pmra_error
#dfx['dr2_pmdec'] = df2.pmdec
#dfx['dr2_pmdec_e'] = df2.pmdec_error

#number of samples 
n_samples = len(dfx.parallax)
imshape = dfx.parallax[0].shape
pm = np.sqrt(dfx.pmra**2+dfx.pmdec**2) 

#data to train on
#data = np.column_stack((dfx.pmra**2+dfx.pmdec**2,dfx.pmra_error**2+dfx.pmdec_error**2,dfx.parallax, dfx.gmag, 
                            #dfx.parallax_over_error, dfx.astrometric_excess_noise, dfx.astrometric_gof_al,))
data = np.column_stack((dfx.pmra-df2.pmra,dfx.pmdec-df2.pmdec,dfx.astrometric_excess_noise,dfx.astrometric_gof_al,
                            dfx.parallax,dfx.parallax_over_error,dfx.pmra,dfx.pmdec,pm),)
nstacks = data.shape[-1]

msk = np.isfinite(data[:,0])  # get rid of infinities/nans
for i in range(1,nstacks):
        msk &= np.isfinite(data[:,i]) 
dfnew = dfx.iloc[msk]
data = data[msk]
data = (data-np.mean(data,axis=0))/np.var(data,axis=0)
#doing this above gives a smaller len(data) value how to fix?

#print(len(df.index))

#pmx = np.sqrt((dfx.pmra)**2 + (dfx.pmdec)**2)
#pm_tot = np.sqrt((dfx.pmra - df2.pmra)**2 + (dfx.pmdec - df2.pmdec)**2)

targ = np.log10(dfnew.chi2acc) #how to subtract pmra and pmdec from both catalogs?
#targ = np.column_stack((dfx.pmra - df2.pmra,dfx.pmdec - df2.pmdec,pm,)) #Does pm matter or not?
#pm does not seem to make a difference

regressor = rndfor(n_estimators=100)

X_train, X_test, y_train, y_test = train_test_split(data[:], targ[:],
                                                   test_size=0.5,shuffle=True)
#X_train = np.nan_to_num(X_train)
#y_train = np.nan_to_num(y_train)
#X_test = np.nan_to_num(X_test)
#y_test = np.nan_to_num(y_test)

n_train = len(y_train)
n_test = len(y_test)

res = regressor.fit(X_train, y_train)

predtrain = regressor.predict(X_train)
predtest = regressor.predict(X_test)

thrval = 250 
accval = 11.8 #chi2 equivilent to 3 sigma
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
#play around with thrval and accval see if you can get better
#mess around with data and see if they make a difference
#next step look for canadiates 
#look into how to query both catalogs in one query
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

sys.exit()
plt.savefig()
    