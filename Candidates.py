#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:33:48 2022

@author: joshhill
"""
#import your stuff
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from astroquery.gaia import Gaia
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn import svm


#call full hgca table
dft = pd.read_csv("/Users/joshhill/Gaia Codes/hgca.csv")

#call data lines of intrest
a_gof = dft.astrometric_gof_al.to_numpy()
a_en = dft.astrometric_excess_noise.to_numpy()
pmra = dft.pmra.to_numpy()
pmdec = dft.pmdec.to_numpy()
pmra_e = dft.pmra_error.to_numpy()
pmdec_e = dft.pmdec_error.to_numpy()
par = dft.parallax.to_numpy()

#number of samples 
n_samples = len(par)
imshape = par[0].shape

#data to train on
n_parms = 7
data = np.zeros((n_samples,n_parms))
data [:,0]= np.nan_to_num(a_gof)
data [:,1]= np.nan_to_num(a_en)
data [:,2]= np.nan_to_num(pmra)
data [:,3]= np.nan_to_num(pmdec)
data [:,4]= np.nan_to_num(pmra_e)
data [:,5]= np.nan_to_num(pmdec_e)
data [:,6]= np.nan_to_num(par)

#target data
targ = np.zeros(n_samples)
targ = np.where((a_gof<5),1,0) #I have no idea on this one look at 

#define classifier
classifier = svm.SVC(gamma = 5)

#split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data[:], targ[:],
                                                    test_size=0.5,shuffle=True)
n_train = len(y_train)
n_test = len(y_test)
classifier.fit(X_train, y_train)

#query
#turn to False after first run
if True:
    query = ("select top 115276 "
                      "parallax, astrometric_gof_al, astrometric_excess_noise, pmra, pmdec, pmra_error, pmdec_error"
                      "from gaiaedr3.gaia_source where parallax_over_error >= 20 AND b > 30 order by random_index")
    job = Gaia.launch_job(query=query)
    r = job.get_results()
    df = r.to_pandas()
    df.to_csv('poe20_b30candidate.csv')
else: 
    df = pd.read_csv('poe20_b30candidate.csv')
    
#data from query

a_gofq = df['astrometric_gof_al'].to_numpy()
a_enq = df['astrometric_excess_noise'].to_numpy()
pmraq = df['pmra'].to_numpy()
pmdecq = df['pmdec'].to_numpy()
pmra_eq = df['pmra_error'].to_numpy()
pmdec_eq = df['pmdec_error'].to_numpy()
parq = df['parallax'].to_numpy()


data1 = np.zeros((n_samples,n_parms))
data1 [:,0]= np.nan_to_num(a_gof)
data1 [:,1]= np.nan_to_num(a_en)
data1 [:,2]= np.nan_to_num(pmra)
data1 [:,3]= np.nan_to_num(pmdec)
data1 [:,4]= np.nan_to_num(pmra_e)
data1 [:,5]= np.nan_to_num(pmdec_e)
data1 [:,6]= np.nan_to_num(par)

newtarg = np.where((a_gof<5),1,0)#i have no idea again

newtest = classifier.predict(data1)
print(np.sum(newtarg==newtest))


cm = confusion_matrix(newtarg, newtest, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classifier.classes_)
disp.plot()
plt.show()

    