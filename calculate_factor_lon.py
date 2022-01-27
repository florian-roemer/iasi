#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 16:37:30 2021

@author: u301023

sbatch_simple_bigmem compute2 factor 36 03:00:00 python calculate_factor.py
"""
import glob
import numpy as np

PATH = '/work/um0878/user_data/froemer/rare_mistral/data/IASI/final/lonlat'

lon = []
for FILE in np.sort(glob.glob(f'{PATH}/lon*', recursive=False)):
    lon.append(np.load(FILE))
lon = np.concatenate(lon, axis=0).flatten() + 180
# %%

hist = np.histogram(lon.flatten(), bins=180, range=(0, 360))

factor = np.mean((hist[0]))/hist[0]
np.save('/work/um0878/user_data/froemer/rare_mistral/data/IASI/final/factor_lon3.npy', factor)
