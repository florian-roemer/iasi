#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:14:27 2022

@author: u301023
"""

import numpy as np
import glob
import matplotlib.pyplot as plt


plt.close('all')

path = '/mnt/lustre02/work/um0878/user_data/froemer/rare_mistral/data/IASI/'\
       'final/lonlat'

iasilat, iasilon, iasicf = [], [], []

for FILE in np.sort(glob.glob(f'{path}/lat*', recursive=False)):
    iasilat.append(np.load(FILE))
    
for FILE in np.sort(glob.glob(f'{path}/lon*', recursive=False)):
    iasilon.append(np.load(FILE))

for FILE in np.sort(glob.glob(f'{path}/cf*', recursive=False)):
    try:
        iasicf.append(np.load(FILE))
    except:
        print(FILE)


ilat = np.concatenate(iasilat, axis=0).flatten()
ilon = np.concatenate(iasilon, axis=0).flatten()
icf = np.concatenate(iasicf, axis=0).flatten()

# %%
factor = np.load('/pf/u/u301023/iasi/factor.npy')
factor_lon = np.load(
    '/work/um0878/user_data/froemer/rare_mistral/data/IASI/final/factor_lon.npy')

# %%
lathist = np.histogram(ilat, bins=180)
lonhist = np.histogram(ilon, bins=360)
cfhist = np.histogram(icf, bins=180)

lathist_cf = np.histogram(ilat[icf==0], bins=180)
lonhist_cf = np.histogram(ilon[icf==0], bins=360)

# %%

path = '/mnt/lustre02/work/um0878/user_data/froemer/rare_mistral/data/CMIP6/MPI-ESM1-2-HR/historical/global/lists'
lat = np.load(f'{path}/latitude_list_n500.npy')
lon = np.load(f'{path}/longitude_list_n500.npy')
cf = np.load(f'{path}/cloudfrac_list_n500.npy')
wap = np.load(f'{path}/wap500_list_n500.npy')
deglat = np.load(f'{path}/deglat.npy')

deglon = np.linspace(-180, 180, 384)

lat = deglat[lat]
lon = deglon[lon]

thresh_cf = 24.497
thresh_wap = 0.068381

lat_cf = lat[cf < thresh_cf]
lon_cf = lon[cf < thresh_cf]

lat_wap = lat[wap > thresh_wap]
lon_wap = lon[wap > thresh_wap]

# %%

model_cf_lat = np.histogram(lat_cf, bins=192)
model_wap_lat = np.histogram(lat_wap, bins=192)

plt.figure(figsize=(15, 6))
plt.step((model_cf_lat[1])[:-1], 100*model_cf_lat[0]/np.sum(model_cf_lat[0]), linewidth=2, 
         label=f'MPI: cloud fraction filter (N={np.sum(model_cf_lat[0])})')
plt.step((model_wap_lat[1])[:-1], 100*model_wap_lat[0]/np.sum(model_wap_lat[0]), linewidth=2, 
         label=f'MPI: omega filter (N={np.sum(model_wap_lat[0])})')
plt.step((lathist_cf[1])[:-1], 100*(lathist_cf[0]*factor)/ np.sum(lathist_cf[0]*factor),
         linewidth=2, label=f'IASI: cloud filter (N={int(np.sum(lathist_cf[0]*factor))})')
plt.xlim(-85, 85)
plt.legend(fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel('latitude', fontsize=18)
plt.ylabel('relative frequency (%)', fontsize=18)

# %%

model_cf_lat = np.histogram(lat_cf, bins=192)
model_wap_lat = np.histogram(lat_wap, bins=192)

plt.figure(figsize=(15, 6))
plt.step((model_cf_lat[1])[:-1], 100*np.cos(np.deg2rad(model_cf_lat[1])[:-1])*model_cf_lat[0]/
         np.sum(np.cos(np.deg2rad(model_cf_lat[1])[:-1])*model_cf_lat[0]), linewidth=2, 
         label=f'MPI: cloud fraction filter (N={np.sum(model_cf_lat[0])})')
plt.step((model_wap_lat[1])[:-1], 100*np.cos(np.deg2rad(model_wap_lat[1])[:-1])*model_wap_lat[0]/
         np.sum(np.cos(np.deg2rad(model_wap_lat[1])[:-1])*model_wap_lat[0]), linewidth=2, 
         label=f'MPI: omega filter (N={np.sum(model_wap_lat[0])})')
plt.step((lathist_cf[1])[:-1], 100*np.cos(np.deg2rad(lathist_cf[1])[:-1])*(lathist_cf[0]*factor)/
         np.sum(np.cos(np.deg2rad(lathist_cf[1])[:-1])*lathist_cf[0]*factor),
         linewidth=2, label=f'IASI: cloud filter (N={int(np.sum(lathist_cf[0]*factor))})')
plt.step((lathist[1])[:-1], 100*np.cos(np.deg2rad(lathist[1])[:-1])*(lathist[0]*factor)/
         np.sum(np.cos(np.deg2rad(lathist[1])[:-1])*lathist[0]*factor),
         linewidth=2, label=f'IASI: cloud filter (N={int(np.sum(lathist_cf[0]*factor))})')
plt.xlim(-85, 85)
plt.legend(fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel('latitude', fontsize=18)
plt.ylabel('area fraction (%)', fontsize=18)

# %% index for subtropics
i1 = np.where((abs(model_cf_lat[1])[:-1] <= 40) & (abs(model_cf_lat[1])[:-1] >= 10))
i2 = np.where((abs(model_wap_lat[1])[:-1] <= 40) & (abs(model_wap_lat[1])[:-1] >= 10))
i3 = np.where((abs(lathist_cf[1])[:-1] <= 40) & (abs(lathist_cf[1])[:-1] >= 10))

rf1 = 100 * np.sum((model_cf_lat[0])[i1]) / np.sum(model_cf_lat[0])
af1 = 100 * np.sum((np.cos(np.deg2rad(model_cf_lat[1])[:-1])*model_cf_lat[0])[i1]) / \
    np.sum(np.cos(np.deg2rad(model_cf_lat[1])[:-1])*model_cf_lat[0])

rf2 = 100 * np.sum((model_wap_lat[0])[i1]) / np.sum(model_wap_lat[0])
af2 = 100 * np.sum((np.cos(np.deg2rad(model_wap_lat[1])[:-1])*model_wap_lat[0])[i1]) / \
    np.sum(np.cos(np.deg2rad(model_wap_lat[1])[:-1])*model_wap_lat[0])

rf3 = 100 * np.sum((lathist_cf[0])[i3]) / np.sum(lathist_cf[0])
af3 = 100 * np.sum((np.cos(np.deg2rad(lathist_cf[1])[:-1])*lathist_cf[0])[i3]) / \
    np.sum(np.cos(np.deg2rad(lathist_cf[1])[:-1])*lathist_cf[0])

# %%
model_cf_lon = np.histogram(lon_cf, bins=384)
model_wap_lon = np.histogram(lon_wap, bins=384)

plt.figure(figsize=(15, 6))
# plt.step((lonhist[1])[:-1], 100*(lonhist[0]*factor_lon)/ np.sum(lonhist[0]*factor_lon),
#           linewidth=2, label='IASI: cloud filter')

plt.step((model_cf_lon[1])[:-1], 100*model_cf_lon[0]/np.sum(model_cf_lon[0]), linewidth=1, 
         label=f'MPI: cloud fraction filter (N={np.sum(model_cf_lon[0])})')
plt.step((model_wap_lon[1])[:-1], 100*model_wap_lon[0]/np.sum(model_wap_lon[0]), linewidth=1, 
         label=f'MPI: omega filter (N={np.sum(model_wap_lon[0])})')
plt.step((lonhist_cf[1])[:-1], 100*(lonhist_cf[0]*factor_lon)/ np.sum(lonhist_cf[0]*factor_lon),
         linewidth=1, label=f'IASI: cloud filter (N={int(np.sum(lonhist_cf[0]*factor_lon))})')
# plt.xlim(-85, 85)
plt.legend(fontsize=16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel('longitude', fontsize=18)
plt.ylabel('relative frequency (%)', fontsize=18)

 # %%
plt.close('all')

thresh_cf = 0
thresh_wap = np.percentile(wap, 95)
 
plt.figure(figsize=(10, 5))
plt.hist2d(lon[cf<=thresh_cf], lat[cf<=thresh_cf], bins=100, cmap='Greys')
plt.yticks(np.arange(-90, 90, 10))
plt.xticks(np.arange(0, 360, 20))

plt.figure(figsize=(10, 5))
plt.hist2d(lon[wap>=thresh_wap], lat[wap>=thresh_wap], bins=100, cmap='Greys')
plt.yticks(np.arange(-90, 90, 10))
plt.xticks(np.arange(0, 360, 20))

cf_thresh = 0
plt.figure(figsize=(10, 5))
plt.hist2d(ilon[icf<=0], ilat[icf<=0], bins=100, range=np.array(([-180,180], [-90, 90])), cmap='Greys')
plt.yticks(np.arange(-90, 90, 10))
plt.xticks(np.arange(0, 360, 20))
plt.title(f'IASI: cloud fraction threshold {cf_thresh}%')

# %%

