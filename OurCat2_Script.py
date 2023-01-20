#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 11:29:57 2022

@author: marcwhiting
"""

import numpy as np
import pandas as pd
import sys, os
import warnings
warnings.filterwarnings("ignore")
import pylab as pl
from matplotlib import cm

from scipy.stats import gaussian_kde 
from scipy.interpolate import interpn
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter
from sklearn.ensemble import RandomForestRegressor as rndfor
from sklearn.tree import DecisionTreeRegressor as dectree

from astropy.table import Table
from astropy.io.votable import from_table, writeto
from astroquery.gaia import Gaia
import os
import sys
#import myfig
import h5py
import sys

np.random.seed(13451345)
np.random.seed(777134134)

#save script to file
rescale = 'raw'
if len(sys.argv)==2:
    rescale = sys.argv[1]
    if not rescale in 'std/var/raw':
        print('error in command line arg.')
        sys.exit()
else:
    print('usage:',sys.argv[0],'raw|std|var')
  

mode = 'accstars'
ok = False


def getHGCA(fname):
    #X = np.genfromtxt(fname,delimiter='|',converters = {6: mycnvtr})
    EDR3lis = np.loadtxt(fname,delimiter='|',dtype=(str),usecols=(1))    
    chi2lis = np.loadtxt(fname,delimiter='|',usecols=(-1))    
    return EDR3lis,chi2lis

def accstar(chi2lin,thresh=30):
    return (chi2lin>thresh)

def accrescale(c2a,inverse=False): 
    if inverse:
        return np.where(c2a>-99,10**c2a,-99)
    else:
        return np.where(c2a>0,np.log10(c2a),-99)

def df2data(df,use_dr2=True, rescale='raw'): # set up, get rid of nans. scale is '', 'std', 'var'
    #df = df.sample(frac=1) This is the one I am curious about 
    
    pmra_cat = df.dr2_pmra
    pmdec_cat = df.dr2_pmdec

    w_x = -.077
    w_y = -.096
    w_z = -.002


    def frameadjust(pmra_cat,pmdec_cat,ra,dec):
        
        #global w_x,w_y,w_z
        deg2rad = np.pi/180
        a = w_x*np.cos(ra*deg2rad)*np.sin(dec*deg2rad)
        b = w_y*np.sin(ra*deg2rad)*np.sin(dec*deg2rad)
        c = w_z*np.cos(dec*deg2rad)
        
        pmra_true = pmra_cat+a+b-c
            
        d = w_x*np.sin(ra*deg2rad)
        e = w_y*np.cos(ra*deg2rad)
        
        
        pmdec_true = pmdec_cat+e-d
        return (pmra_true, pmdec_true)


    df.dr2_pmra, df.dr2_pmdec = frameadjust(pmra_cat,pmdec_cat,df.ra,df.dec)
    
    
    #nstacks = 5
    #data = np.column_stack((df.bp_rp, df.gmag, df.pmra, df.pmdec, df.parallax,df.astrometric_gof_al))
    if not use_dr2:
        data = np.column_stack((\
                                np.sqrt(df.pmra**2+df.pmdec**2), 
                                np.sqrt(df.pmra_error**2+df.pmdec_error**2), 
                                #np.sqrt((df.dr2_pmdec-df.pmdec)**2+(df.dr2_pmra-df.pmra)**2),
                                df.pmra, df.pmdec,
                                df.pmra_error, df.pmdec_error, 
                                #df.dr2_pmra-df.pmra,df.dr2_pmdec-df.pmdec,
                                df.ruwe,
                                df.parallax, df.parallax_over_error,
                                df.phot_g_mean_mag, 
                                df.bp_rp, 
                                df.astrometric_excess_noise_sig,
                                df.astrometric_gof_al))
    else:
        data = np.column_stack((\
                                np.sqrt(df.pmra**2+df.pmdec**2), 
                                np.sqrt(df.pmra_error**2+df.pmdec_error**2), 
                                np.sqrt((df.dr2_pmdec-df.pmdec)**2+(df.dr2_pmra-df.pmra)**2),
                                df.pmra, df.pmdec,
                                df.pmra_error, df.pmdec_error, 
                                df.dr2_pmra-df.pmra,df.dr2_pmdec-df.pmdec,
                                df.ruwe,
                                df.parallax, df.parallax_over_error,
                                df.phot_g_mean_mag, 
                                df.bp_rp, 
                                df.astrometric_excess_noise_sig,
                                df.astrometric_gof_al))
    if False:
        pm = np.sqrt(df.pmra**2+df.pmdec**2)
        if use_dr2:        
            data = np.column_stack((df.pmra-df.dr2_pmra,df.pmdec-df.dr2_pmdec,
                                    df.astrometric_gof_al,df.parallax, df.parallax_over_error,df.pmra, df.pmdec, pm))
        else:
            data = np.column_stack((df.astrometric_gof_al,df.parallax, df.parallax_over_error,df.pmra, df.pmdec, pm))
        '''
        n_samples = len(dfrndx.parallax)% gives us how many stars are in the random catalog
imshape = dfrndx.parallax[0].shape% this tells us the shape of the dataframe
pmrnd = np.sqrt(dfrndx.pmra**2+dfrndx.pmdec**2)%total proper motion of random catalog in edr3
pmrnd3 = np.sqrt(dfrnd3.pmra**2+dfrnd3.pmdec**2)%Ignore this
pm = np.sqrt(dfx.pmra**2+dfx.pmdec**2)% total proper motion of HGCA in edr3

%This is the parameters that we feed into the random forest
data = np.column_stack((dfx.pmra-df2.pmra,dfx.pmdec-df2.pmdec,dfx.astrometric_gof_al,dfx.parallax,dfx.parallax_over_error, dfx.pmra,dfx.pmdec,pm),)%This one is for the HGCA
datarnd = np.column_stack((dfrndx.pmra-dfrnd2.pmra,dfrndx.pmdec-dfrnd2.pmdec,dfrndx.astrometric_gof_al,dfrndx.parallax,dfrndx.parallax_over_error,dfrndx.pmra,dfrndx.pmdec,pmrnd),)%This is for the random catalog
        '''
    
    
    print('df2data:',len(data))
    nstacks = data.shape[-1]
    msk = np.isfinite(data[:,0])
    print('0',np.sum(msk))
    for i in range(1,nstacks):
        msk &= np.isfinite(data[:,i])
    dfnew = df.iloc[msk]
    data = data[msk]
    if rescale == 'var': 
        data = (data-np.mean(data,axis=0))/np.var(data,axis=0)
        print('data rescale=var')
    elif rescale == 'std':
        data = (data-np.mean(data,axis=0))/np.std(data,axis=0)
        print('data rescale=std')
    elif rescale != 'raw':
        print('rescale error...')
        sys.exit()
    labels = accstar(dfnew.chi2acc)
    values = accrescale(dfnew.chi2acc)
    return dfnew,data,values,labels


def get_dens(X,Y,logx=False,logy=False,nbins=200,returngrid=False):
    msk = np.isfinite(X) & np.isfinite(Y)
    x,y = np.copy(X[msk]),np.copy(Y[msk])
    if logx: x = np.log(x)
    if logy: y = np.log(y)
    if len(x)>20000 or returngrid:
        h2d,xe,ye = np.histogram2d(x, y, bins = nbins)
        if returngrid:
            xg,yg = xe[1:]+0.5*(xe[1]-xe[0]),ye[1:]+0.5*(ye[1]-ye[0])
            if logx: xg = np.exp(xg)
            if logy: yg = np.exp(yg)
            print((xg[0]),(xg[-1]),(yg[0]),(yg[-1]))
            return xg,yg,h2d
        z = interpn((0.5*(xe[1:]+xe[:-1]),0.5*(ye[1:]+ye[:-1])),h2d,np.vstack([x,y]).T, method = "splinef2d",bounds_error=False)
    else:
        sca = 0.05
        qq = np.vstack([x,y])
        z = gaussian_kde(qq,bw_method=sca)(qq)
    idx = np.argsort(z)
    #print(qq.shape,type(qq),z.shape)
    x,y,z = x[idx],y[idx],z[idx]
    if logx: x = np.exp(x)
    if logy: y = np.exp(y)
    return x,y,z

def confuse(predict,true,nlabels):
    print('true:')
    for i in range(nlabels):
        print('%2d   |  '%(i),end='')
        for j in range(nlabels):
            predtrue = np.sum((predict == j) & (true==i))
            print(' %5d'%(predtrue),end='')
        print(' ')    
    print('-----+----------------------')
    print('predicted:',end='')
    for i in range(nlabels): print('  %d  '%(i),end='')
    print(' ')

c = 'k'

def mergedr2edr3(df2,df3):
    msk = np.isin(df3.source_id,df2.dr3_source_id)
    dfx = df3[msk].copy() # ijust in case?
    dfx.sort_values(by='source_id',inplace = True)
    dfx.reset_index(drop=True,inplace=True)
    df2.sort_values(by='dr3_source_id',inplace=True)
    df2.reset_index(drop=True,inplace=True)
    dfx['dr2_source_id'] = df2.source_id;
    dfx['dr2_pmra'] = df2.pmra
    #dfx['dr2_pmra_error'] = df2.pmra_error;
    dfx['dr2_pmdec'] = df2.pmdec
    #dfx['dr2_pmdec_error'] = df2.pmdec_error
    print('total dr2:',len(df2.index))
    print('edr3 w/dr2 srcs identified:',len(dfx.index))
    print('id cross check (0 is good):',np.sum(dfx.source_id - df2.dr3_source_id))
    return dfx

def getnssmatch(facc,dfx):
    # bring in the DR3 accelerating catalog
    df3a = pd.read_pickle(facc)
    print('loaded nss stars:',len(df3a.index))
    print('min sig',np.min(df3a.significance))
    print(' naccelsolntype=7',np.sum(df3a.nss_solution_type.str.contains('Acceleration7')))
    #if ('nss_solution_type' in df3a.columns.tolist()):
    #    msk = df3a.nss_solution_type.str.contains('Orbital') | df3a.nss_solution_type.str.contains('Acceleration')
    #    msk = df3a.nss_solution_type.str.contains('Orbital') | df3a.nss_solution_type.str.contains('Acceleration')
    #    df3a = df3a[msk].copy()
    #    print('selected orbit solutions, dr3acc:',len(df3a.index))
    #else:
    #    print('ok')
    #sys.exit()
    #xmatch = np.isin(dfx.source_id,df3a.source_id)
    #dfxa = dfx[xmatch].copy()
    #print('dfx-nss matches:',len(dfxa.index))
    xmatch = np.isin(df3a.source_id,dfx.source_id)
    dfax = df3a[xmatch].copy()
    #print('nss-dfx matches:',len(dfax.index)) # wtf is going in here?
    uu,ct = np.unique(dfax.source_id,return_counts=True)
    if np.sum(ct>1):
        dfax.drop_duplicates(subset=['source_id'],inplace=True)
    dfax.sort_values(by='source_id',inplace = True)
    dfax.reset_index(drop=True,inplace=True)
    print('nss-dfx matches:',len(dfax.index)) # wtf is going in here?
    return dfax #,dfxa #ax is accel, xa is hgca x-matched to accel




fdr3 = '/Users/marcwhiting/zip/hipgedr3.zip'
fdr2 = '/Users/marcwhiting/zip/hipgdr2.zip'
flis = '/Users/marcwhiting/dat/catalog.dat' # ascii data, contains a list of source_ids from edr3.

df3 = pd.read_pickle(fdr3)
df2 = pd.read_pickle(fdr2)

elis,chi2lis = getHGCA(flis) # elis is the edr3 source_id, as a string, chi2lis is some other thing we use here...
elis = elis.astype(np.int64) # this is the source_id as an int

#probably ref frame here 

chksum = np.sum((df3.source_id - elis)**2)
print('HGCA+gaia edr3 chksum to ensure ordering of source_ids (0 is good):',np.sum(df3.source_id != elis))
df3['chi2acc'] = chi2lis

dfx = mergedr2edr3(df2,df3)

#print (dfx.source_id)
#sys.exit()

#print("souce id 1:", dfx['source_id'].iloc[0])

#sys.exit()

#msk = (dfx.parallax > 10) & ((dfx.b > 20) | (dfx.b < -20))
#dfxcut =  dfx[msk].copy()
print('train+test HGCA:',len(dfx.index))#,len(dfxcut.index))

cmap=cm.inferno
siz=1.0; 

regressor = rndfor(n_estimators=150)

testtrain = True
#testtrain = False

if testtrain:
    tablelines = ['']*7
    for use_dr2 in [True, False]:
        dfx,data,values,labels = df2data(dfx,use_dr2=use_dr2,rescale=rescale)
        print(data[0])
        sys.exit()
        ndata = len(dfx.index); nparms = data.shape[1]
        ntrain = 4*ndata//8; ntest = ndata-ntrain
        Xtrain = data[:ntrain]; ytrain = values[:ntrain]
        Xtest = data[ntrain:];  ytest = values[ntrain:]
        print('number in train set:',ntrain)
        #regressor = dectree() # n_estimators=100)
        res = regressor.fit(Xtrain, ytrain)
        print('done fit.',res)
        predtrain = regressor.predict(Xtrain)
        print('compare...')
        #[print(y,p) for [y,p] in list(zip(ytrain,predtrain))[0:5]]
        # check train
        print('rms err for training set',np.std(ytrain-predtrain))
        # run test
        predtest = regressor.predict(Xtest)
        print('rms err for test set',np.std(ytest-predtest))
        dfx['chi2accpredicted'] = accrescale(regressor.predict(data),inverse=True)
        dfx['labelpredicted'] = accstar(dfx.chi2accpredicted)

        threshvals = np.logspace(np.log10(11),np.log10(1000),20)
        ndata = len(dfx.index)

        accval = 28.75
        chi2test = 10**ytest
        chi2accpredicted = 10**predtest
        print('accval:',accval,'unambiguously acc',np.sum((chi2test>=accval)))
        print('threshval, accuracy efficiency ')
        # for thrval in [11.8, 28.75, 50, 100,250,500,1000,2500]:
        
        for i, thrval in enumerate([28.75, 50, 100,250,500,1000,2500]):
            groundtrupos = np.sum((chi2test>=accval))
            groundtruneg = np.sum((chi2test<accval))
            trupos = np.sum((chi2test>=accval)&(chi2accpredicted>=thrval))
            obspos = np.sum((chi2accpredicted>=thrval))
            #print(' threshval, accval:',thrval,accval)
            #print('     groundtru pos',groundtrupos)
            #print('     groundtru neg',groundtruneg)
            #print('     trupos,obspos,trupos/obspos:',trupos,obspos,trupos/obspos)
            dl,uu = r'&',r'\%'
            if use_dr2:
                tablelines[i] = ' %g & %3.1f%s & %3.1f%s & '%(thrval,100*trupos/obspos,uu,100*obspos/groundtrupos,uu)
            else:
                tablelines[i] += ' %3.1f%s & %3.1f%s \\\\'%(100*trupos/obspos,uu,100*obspos/groundtrupos,uu)
            #print(tablelines[i])
        if use_dr2 == False:
            print(r'% thresh % w/dr2: trupos/obspos_this_thresh obspos/trupos_total, same for w/o dr2....')
            for line in tablelines:
                print(line)


fedr3xx = 'edr3rndchi2acc'+rescale+'.zip'

predict_catacc = True
#predict_catacc = False

maglim = 17.5

if predict_catacc:
    # work with new cat data 'rnd' is an old name...
    # => a bunch of new stars 
    fdr3rnd = '/Users/marcwhiting/zip/edr3rnd.zip'
    df3rnd = pd.read_pickle(fdr3rnd)
    df3rnd['chi2acc'] = -1 * np.ones(len(df3rnd.index))
    fdr2rnd = '/Users/marcwhiting/zip/dr2rnd.zip'
    df2rnd = pd.read_pickle(fdr2rnd)
    dfxrnd = mergedr2edr3(df2rnd,df3rnd)
    
    #print(dfxrnd.columns)
    dfxrnd = dfxrnd[dfxrnd.phot_g_mean_mag < maglim].copy() # copy necessary?
    print('# total num stars in our cat, gmag < '+str(maglim)+', in DR2 and EDR3:',len(dfxrnd.index))
    msk = np.isin(dfxrnd.source_id,dfx.source_id)
    dfxrnd = dfxrnd[~msk].copy()
    print('# total num stars in our cat w/o HGCA:',len(dfxrnd.index))
    
    # fuck...just in case...something went wrong downstream...
    dfxrnd.sort_values(by='source_id',inplace = True)
    dfxrnd.reset_index(drop=True,inplace=True)

    print('total new stars:',len(dfxrnd.index),np.sum(~msk))
    print('check:',np.sum(np.isin(dfx.source_id,dfxrnd.source_id)),'should be 0')
    
    # merge in our cat and nss_acc....
    df3ax = getnssmatch('/Users/marcwhiting/zip/dr3acc.zip',dfxrnd)
    # df3ax is source_id sorted.. dfxrnd is too
    xmatch = np.isin(dfxrnd.source_id,df3ax.source_id)
    msk = dfxrnd.source_id[xmatch].to_numpy() != df3ax.source_id.to_numpy()
    if np.sum(msk):
        print('xmatch failure!!!!!',np.sum(np.abs(dfxrnd.source_id[xmatch] - df3ax.source_id)))
        quit()
    dfxrnd['nss_accel_significance'] = 0.0
    dfxrnd.nss_accel_significance[xmatch] = df3ax.significance.to_numpy() #df3ax.source_id//10000
    #print(np.sum(dfxrnd.nss_accel_significance>0.0),len(df3ax.index))
    nnssacc = len(df3ax.index)

    # merge in our cat and nss_twobody....
    dfxrnd.sort_values(by='source_id',inplace = True)
    dfxrnd.reset_index(drop=True,inplace=True)
    df32b = getnssmatch('/Users/marcwhiting/zip/dr3twobody.zip',dfxrnd)
    xmatch = np.isin(dfxrnd.source_id,df32b.source_id)
    #print('zzz',np.sum(xmatch),len(df32b.index))
    msk = dfxrnd.source_id[xmatch].to_numpy() != df32b.source_id.to_numpy()
    print(np.sum(msk))
    if np.sum(msk):
        print('xmatch failure!!!!!',np.sum(np.abs(dfxrnd.source_id[xmatch] - df32b.source_id)))
        quit()
    dfxrnd['nss_twobody_significance'] = 0.0
    dfxrnd.nss_twobody_significance[xmatch] = df32b.significance.to_numpy() #df3ax.source_id//10000

    # all data on board....
    print('data set minus HGCA:',len(dfxrnd))
    use_dr2 = True
    dfxrnd,datarnd,valuesrnd,labelsrnd = df2data(dfxrnd,use_dr2=use_dr2,rescale=rescale)
    print('going into training, # of cat sources:',len(dfxrnd))
    
    # take the classifier out for a spin!
    print('retraining, with full HGCA!')
    use_dr2 = True
    dfx,data,values,labels = df2data(dfx,use_dr2=use_dr2,rescale=rescale)
    ndata = len(dfx.index); nparms = data.shape[1]
    res = regressor.fit(data, values)
    print('done training.')

    print('predicting...')
    predrnd = regressor.predict(datarnd)
    dfxrnd['chi2accpredicted'] = accrescale(predrnd,inverse=True)


    print('total new stars tested:',len(dfxrnd.index))


    print('writing to',fedr3xx)
    dfxrnd.to_pickle(fedr3xx)

if not predict_catacc:
    # then read in the data....
    if os.path.isfile(fedr3xx):
        dfxrnd = pd.read_pickle(fedr3xx)
        print('read',len(dfxrnd.index),'sources from',fedr3xx)
    else:
        print("can't find cat file", fedr3xx, "w/predicted accs, etc")
        quit()

if True:
    nnssax = np.sum(dfxrnd.nss_accel_significance > 0.0)
    nnss2b = np.sum(dfxrnd.nss_twobody_significance > 0.0)
    print('nss_accel stars:',nnssax)
    print('nss_2body stars:',nnss2b)
    print(' in both:',np.sum((dfxrnd.nss_accel_significance > 0.0)&(dfxrnd.nss_twobody_significance > 0.0)))
    ncat = len(dfxrnd.index)
    # analyze the cat data....
    print(r'% thresh, Ncat, Ncat/Ntot, Ncatacc, Ncatacc/Nacc, Ncat2bdy Ncat2bdy/N2bdy')
    for thrval in [11.8,28.75,100,250,500,1000]:
        msk = dfxrnd.chi2accpredicted > thrval 
        mskx = msk & (dfxrnd.nss_accel_significance > 0.0)
        msk2 = msk & (dfxrnd.nss_twobody_significance > 0.0)
        #print(' threshval,nstars, nstars_in_nss_acc:',thrval,np.sum(msk),np.sum(mskx))
        #print(' threshval,nstars, nstars_in_nss_acc:',thrval,np.sum(msk),np.sum(mskx))
        ncatthis,nnssaxthis,nnss2bthis = np.sum(msk),np.sum(mskx),np.sum(msk2)
        uu = r'\%'
        print('%4g & %d  & %4.1f%s & %d & %4.1f%s & %d & %4.1f%s \\\\'%\
              (thrval,ncatthis,ncatthis/ncat*100,uu,nnssaxthis,100*nnssaxthis/nnssax,uu,nnss2bthis,100*nnss2bthis/nnss2b,uu))

        






