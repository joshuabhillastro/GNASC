#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:25:43 2022

@author: joshhill
"""
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import colors
import pandas as pd

plt.style.use('default')


Table = pd.read_csv("/Users/joshhill/Gaia Data/HGCA_Accel.csv")

r = pd.DataFrame(Table)

bp_rp = r['bp_rp']
M_g = r['phot_g_mean_mag']
Par = r['parallax']
chi2 = r['chi2']
#print(chi2)
d = 1/Par
D = d*1000
M_G = M_g - 5*np.log10(D/10)

h = plt.hist2d(bp_rp, M_G, bins=300, cmin=2, range = [[-1,8],[-10,15]], norm=colors.PowerNorm(0.5), zorder=0.5)
plt.scatter(bp_rp,M_G,s=.5, color='k', zorder=0)
plt.ylim(15,-10)
plt.xlabel('bp-rp')
plt.ylabel('M_g')
cb = plt.colorbar(h[3], ax=plt.subplot(), pad=0.02)
cb.set_label('Stellar Density')

