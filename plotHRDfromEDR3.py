#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 18:42:38 2022

@author: joshhill
"""
from astroquery.gaia import Gaia
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np

Gaia.ROW_LIMIT = -1 #no limit on rows in gaia data
Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source" # Select early Data Release 3

coord = SkyCoord(ra=45, dec=30, unit=(u.degree, u.degree), frame='icrs') #certain part of sky to query
width = u.Quantity(1, u.deg) #how wide to make the query area
height = u.Quantity(1, u.deg) #how high to make the query area
r = Gaia.query_object_async(coordinate=coord, width=width, height=height) #query EDR3

bp_rp = r['bp_rp'].data
M_g = r['phot_g_mean_mag'].data
Par = r['parallax'].data
d = 1/Par
D = d*1000
M_G = M_g - 5*np.log10(D/10)


plt.xlabel('bp-rp')
plt.ylabel('M_g')
plt.scatter(bp_rp,M_G,s=.5)
plt.ylim(15, -10)
