#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:37:30 2021

@author: u301023

sbatch_simple_bigmem compute2 factor 36 03:00:00 python calculate_factor.py
"""
import glob
import numpy as np
import process_iasi as pi

PATH = '/work/um0878/data/iasi/iasi-l1/reprocessed/m02/2013/09/'

LIST = np.sort(glob.glob(PATH + '**/IASI*', recursive=False))[0:412:10]

#hist = np.zeros(shape=(180,), dtype='int')
hist = np.zeros(shape=(len(LIST), 180), dtype='int')

for f, FILE in enumerate(LIST):
    print(FILE)
    data = pi.read_data(FILE)
    lat = data.lat_orbit
#    hist += np.histogram(lat.flatten(), bins=180, range=(-90, 90))[0]
    hist[f, :] = np.histogram(lat.flatten(), bins=180, range=(-90, 90))[0]

np.save('/work/um0878/user_data/froemer/rare_mistral/data/IASI/factor/'
        'histogram.npy', hist, allow_pickle=False)

# %%

hist = np.load('/work/um0878/user_data/froemer/rare_mistral/data/IASI/factor/'
               'histogram.npy')

factor = np.sum(hist, axis=0)[90:92].mean()/np.sum(hist, axis=0)
np.save('/pf/u/u301023/iasi/factor.npy', factor, allow_pickle=False)
