#!/Users/joshhill/opt/bin python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 15:35:52 2022

@author: joshhill
"""
#from astropy.io import ascii
from astropy.table import Table
from astroquery.gaia import Gaia
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import colors
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

query = "select top 115345 source_id,bp_rp,phot_g_mean_mag,parallax,col0 from gaiadr2.gaia_source inner join user_jhill01.source_ids_3 on source_id = col0"
job = Gaia.launch_job(query=query)

r = job.get_results()

Par = r['parallax'].data
d = 1/Par
D = d*1000
msk = (D>0)&(D<100000) 
bp_rp = r['bp_rp'].data
bp_rp = bp_rp[msk]
M_g = r['phot_g_mean_mag'].data
M_g = M_g[msk]
Par = Par[msk]
d = d[msk]
D = D[msk] 
M_G = M_g - 5*np.log10(D/10)

h = plt.hist2d(bp_rp, M_G, bins=300, cmin=2, range = [[-4,8],[-10,15]], norm=colors.PowerNorm(0.5), zorder=0.5)
plt.scatter(bp_rp,M_G,s=.5, color='k', zorder=0)
plt.ylim(15,-10)
plt.xlabel('bp-rp')
plt.ylabel('M_g')
cb = plt.colorbar(h[3], ax=plt.subplot(), pad=0.02)
cb.set_label('Stellar Density')
plt.savefig("HGCA_HRD_Updated.pdf", format="pdf", bbox_inches="tight")




