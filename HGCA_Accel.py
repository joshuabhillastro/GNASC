#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:25:43 2022

@author: joshhill
"""
from astropy.table import Table
from astroquery.gaia import Gaia
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

plt.style.use('default')
#source_id_list = ascii.read("/Users/joshhill/Gaia Data/catalog.dat", delimiter = "|")

source_id_list = pd.read_table("/Users/joshhill/Gaia Data/catalog.dat", delimiter = "|", usecols = [1],dtype = np.int64)

source_id_table = source_id_list.to_numpy()

#source_id_table = Table(source_id_tablenp, names = 'source_id')

#print(source_id_table)

Gaia.login(user='jhill01',password='Supernova1!')

#send_source_id_table = Gaia.upload_table(upload_resource=source_id_table,table_name='source_ids_4')
table_source_id = 'user_jhill01.source_ids_3'

query = "select top 115345 source_id,pm,pmra,pmdec,bp_rp,phot_g_mean_mag,parallax,col0 from gaiaedr3.gaia_source inner join user_jhill01.source_ids_3 on source_id = col0 "
job = Gaia.launch_job(query=query)

r = job.get_results()

bp_rp = r['bp_rp'].data
M_g = r['phot_g_mean_mag'].data
Par = r['parallax'].data
pm = r['pm'].data
pmra = r['pmra'].data
pmdec = r['pmdec'].data
d = 1/Par
D = d*1000
M_G = M_g - 5*np.log10(D/10)
a = pm/((2015.5-1991.25)/2)
t = (3*2015.5+1991.25)/4
a_table = pd.DataFrame(a)
#a_table.to_csv('~/Gaia\ Data')
#print(a_table)

