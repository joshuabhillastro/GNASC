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
import pandas as pd

#source_id_list = ascii.read("/Users/joshhill/Gaia Data/catalog.dat", delimiter = "|")

source_id_list = pd.read_table("/Users/joshhill/Gaia Data/catalog.dat", delimiter = "|", usecols = [1],dtype = np.int64)

source_id_table = Table.from_pandas(source_id_list)

#print(source_id_table)

Gaia.login(user='jhill01',password='Supernova1!')

send_source_id_table = Gaia.upload_table(upload_resource=source_id_table,table_name='source_ids_2')
table_source_id = 'user_jhill01.source_ids_2'

query = "select top 115345 source_id,bp_rp,phot_g_mean_mag,parallax from gaiadr2.gaia_source, source_ids_2" 
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
