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

Par = r['parallax']
d = 1/Par
D = d*1000
msk = (D>0)&(D<100000) 
bp_rp = r['bp_rp']
bp_rp = bp_rp[msk]
M_g = r['phot_g_mean_mag']
M_g = M_g[msk]
Par = Par[msk]
d = d[msk]
D = D[msk] 
M_G = M_g - 5*np.log10(D/10)

h = plt.hist2d(bp_rp, M_G, bins=300, cmin=2, range = [[-1,8],[-10,15]], norm=colors.PowerNorm(0.5), zorder=0.5)
plt.scatter(bp_rp,M_G,s=.5, color='k', zorder=0)
plt.ylim(15,-10)
plt.xlabel('bp-rp')
plt.ylabel('M_g')
cb = plt.colorbar(h[3], ax=plt.subplot(), pad=0.02)
cb.set_label('Stellar Density')

