import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from astroquery.gaia import Gaia
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn import svm
#from sklearn.svm import LinearSVC
#from sklearn.ensemble import BaggingClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import SGDClassifier
#import astropy.units as u
#from astropy.coordinates import SkyCoord
#from astropy.io.votable import parse_single_table
#from matplotlib import colors
#from sklearn.datasets import make_classification
import sys
#sys.exit("Terminate")

dft = pd.read_csv("/Users/marcwhiting/hgca.csv")

#Building in parameters


bp_rpt = dft.bp_rp.to_numpy()
M_gt = dft.gmag.to_numpy()
part = dft.parallax.to_numpy()



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
imshape = part[0].shape 


n_parms = 3
data = np.zeros((n_samples,n_parms))
data [:,0]= np.nan_to_num(part)
data [:,1]= np.nan_to_num(M_gt)
data [:,2]= np.nan_to_num(bp_rpt)

targ = np.zeros(n_samples)
targ = np.where((M_Gt>0.0)&(M_Gt<1.2)&(bp_rpt>1.1)&(bp_rpt<1.4),1,0)



classifier = svm.SVC(gamma = 5)
#classifier = svm.SVC()


#classifier = BaggingClassifier(KNeighborsClassifier())

X_train, X_test, y_train, y_test = train_test_split(data[:], targ[:],
                                                    test_size=0.5,shuffle=True)
n_train = len(y_train)
n_test = len(y_test)
classifier.fit(X_train, y_train)

if True:
    query = ("select top 115 * "
                      "from gaiaedr3.gaia_source where parallax_over_error > 10 AND b > 30"  
                      "AND astrometric_gof_al > 2 "
                      "AND astrometric_excess_noise_sig > 2 order by random_index")
                      
    job = Gaia.launch_job(query=query)
    r = job.get_results()
    df = r.to_pandas()
    df.to_hdf('poe20_b30.h5','data', mode ='w')
else: 
    df = pd.read_hdf('poe20_b30.h5')



sys.exit("Terminate")

#Creating M_G to use with ML


bp_rp = df['bp_rp'].to_numpy()
M_g = df['phot_g_mean_mag'].to_numpy()
Par = df['parallax'].to_numpy()
d = 1/Par
D = d*1000 
M_G = M_g - 5*np.log10(D/10)


n_samples = len(Par)
data1 = np.zeros((n_samples,n_parms))
data1 [:,0]= np.nan_to_num(Par)
data1 [:,1]= np.nan_to_num(M_g)
data1 [:,2]= np.nan_to_num(bp_rp)
#data1 [:,4]= np.nan_to_num(a_gof)

newtarg = np.where((M_G>0.0)&(M_G<1.2)&(bp_rp>1.1)&(bp_rp<1.4),1,0)



newtest = classifier.predict(data1)
print(np.sum(newtarg==newtest))


cm = confusion_matrix(newtarg, newtest, labels=classifier.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=classifier.classes_)
disp.plot()
plt.show()



#now create a CMD
#h = plt.hist2d(bp_rp, M_G, bins=300, cmin = 2,  range = [[-4,8],[-3,20]],
                                   #norm=colors.PowerNorm(0.5), zorder=2.5)
#plt.scatter(bp_rp,M_G,s=.5, color='k', zorder=0)
plt.scatter(bp_rp,M_G,s=.5, color='#aaaaaa', zorder=0)
msk = newtarg==1
plt.scatter(bp_rp[msk],M_G[msk],s=.5, color='k', zorder=1)
msk = newtest==1
plt.scatter(bp_rp[msk],M_G[msk],s=.5, color='b', zorder=2)
plt.ylim(20,-3)
plt.xlim(-4,8)
plt.xlabel('bp-rp')
plt.ylabel('M_G')
#cb = plt.colorbar(h[3], ax=plt.subplot(), pad=0.02)
#cb.set_label('Stellar Density')
