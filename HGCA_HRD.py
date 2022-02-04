#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:35:52 2022

@author: joshhill
"""
from astropy.io import ascii
from astropy.table import Table
from astroquery.gaia import Gaia
import numpy as np
import matplotlib.pyplot as plt

source_id_list = ascii.read("/Users/joshhill/Gaia Data/catalog.dat", delimiter = "|")

source_id_table = Table(source_id_list)

Gaia.login(user='jhill01',password='Supernova1!')

#source_id_table = Gaia.upload_table(upload_resource=source_id_list,table_name='source id')
table_source_id = 'user_jhill01.source id'

query = "select top 120000 source_id,ra,dec,bp_rp,phot_g_mean_mag,parallax from gaiadr2.gaia_source where " 
job = Gaia.launch_job(query=query)

r = job.get_results()

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
