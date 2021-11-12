#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 10:07:08 2021

@author: u301023
"""

import glob
import numpy as np
import time

import process_iasi as pi
import matplotlib.pyplot as plt

start = time.process_time()

#    year = sys.argv[1]
#    month = sys.argv[2]
#    day = sys.argv[3]

year, month, day = '2013', '03', '01'

PATH = f'/work/um0878/data/iasi/iasi-l1/reprocessed/m02/'\
       f'{year}/{month}/{day}/'

for FILE in np.sort(glob.glob(PATH + 'IASI*', recursive=False))[0:1]:

    # read IASI data
    idata = pi.read_data(FILE)

    # assign
    orbit = FILE[74:105]
    radiance = idata.spectra_orbit * 100  # convert units to W cm m-2 sr-1
    lat = idata.lat_orbit
    angle = idata.GGeoSondAnglesMETOP[:, :, :, 0]
    cloud_frac = idata.GEUMAvhrr1BCldFrac[:, :, :]
    land_frac = idata.GEUMAvhrr1BLandFrac[:, :, :]

# %%
plt.close('all')
raw = np.histogram(lat.flatten(), bins=180)
cos_weighted = raw[0]*np.cos(np.deg2rad(np.arange(-90, 90)))
cos = np.cos(np.deg2rad(np.arange(-90, 90)))
lats = np.arange(-90, 90)

plt.plot(lats, raw[0]/cos_weighted.max(), label='no weights')
plt.plot(lats, cos_weighted/cos_weighted.max(), label='cosine weighted')
plt.plot(lats, cos,
         label='pure cosine (actual area)')
plt.vlines(-30, 0, raw[0].max()/cos_weighted.max())
plt.vlines(30, 0, raw[0].max()/cos_weighted.max())
plt.xticks(np.arange(-90, 92, 30))
plt.ylim(0, raw[0].max()/cos_weighted.max())
plt.legend(loc='upper center')

plt.figure()
factor = cos/(cos_weighted/cos_weighted[90])
factor2 =  raw[0][90]/raw[0]
plt.plot(lats, raw[0][90]/raw[0])
plt.plot(lats, cos)
plt.plot(lats, cos*raw[0][90]/raw[0])
#plt.scatter(lats, factor, s=8)
plt.hlines(1, -90, 90)
plt.xticks(np.arange(-90, 92, 30))
plt.ylim(0,2)


def get_scaling_factor(lat):
    hist = np.histogram(lat.flatten(), bins=180)
    factor =  hist[0][90]/hist[0]
    index = lat.astype('int') + 90
    return factor[index]
    

weights = get_scaling_factor(lat)
plt.figure()
plt.hist(weights.flatten(), bins=1000)