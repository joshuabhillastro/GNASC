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
from matplotlib import colors
import pandas as pd
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier


table = pd.read_csv("/Users/joshhill/Gaia Data/HGCA_Accel.csv", usecols = [1,2,3])
bp_rpt = table.iloc[:,0]#check to make sure
M_gt = table.iloc[:,1]
part = table.iloc[:,2]
dt = 1/part
Dt = dt*1000
M_Gt = M_gt -5*np.log10(Dt/10)

#For now I will use parallax for what the practice lab uses as digits
n_samples = len(part) 
imshape = part[0].shape #not sure about this line

n_parms = 2
data = np.zeros((n_samples,n_parms))
#data [:,0]= np.nan_to_num(part)
data [:,0]= np.nan_to_num(M_gt)
data [:,1]= np.nan_to_num(bp_rpt)

targ = np.zeros(n_samples)
targ = np.where((M_Gt<4)&(bp_rpt>1.6),1,0)


# choose your classifier. I stuck with the one from the lab, but this is tbd
#classifier = BaggingClassifier(KNeighborsClassifier())
classifier = svm.SVC()
#classifier = svm.LinearSVC()
#classifier = SGDClassifier()


# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data[:], targ[:],
                                                    test_size=0.5,shuffle=True)
n_train = len(y_train)
n_test = len(y_test)



#EDR3 source ID
query = ("select top 31123 "
                      "solution_id,astrometric_n_obs_al,astrometric_gof_al,astrometric_primary_flag,"
                      "astrometric_excess_noise,phot_g_mean_mag,parallax_error,parallax_over_error,"
                      "bp_rp,phot_g_mean_mag,parallax, "
                      "duplicated_source "
                      "from gaiaedr3.gaia_source where parallax > 1 order by random_index")
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

data1 = np.zeros((n_samples,n_parms))
#data1 [:,0]= np.nan_to_num(Par)
data1 [:,0]= np.nan_to_num(M_G)
data1 [:,1]= np.nan_to_num(bp_rp)



X_train1, X_test1, y_train1, y_test1 = train_test_split(data[:], targ[:],
                                                    test_size=0.5,shuffle=True)
classifier.fit(X_train, y_train)
test = classifier.predict(data1[15561:,:])
#test = classifier.predict(X_test1)
print(np.sum(targ==1))


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
