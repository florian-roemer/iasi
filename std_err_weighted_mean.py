#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:29:05 2021

@author: u301023
"""

import numpy as np
import matplotlib.pyplot as plt

chan = np.load('/work/um0878/user_data/froemer/rare_mistral/data/chan',
               allow_pickle=True)

path = '/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/test'

used = [0, 1, 3, 4, 9, 10, 12, 13, 18, 19, 21, 22]
domains = np.load(f'{path}/domains.npy')[used]
months = np.load(f'{path}/months.npy')
mx = 0
plt.figure()
for d in domains:
    for m in months:
        flux = np.load(f'{path}/{d}/{m}/avg_flux.npy')
        flux2 = np.load(f'{path}/{d}/{m}/avg_flux_squared.npy')
        area = np.load(f'{path}/{d}/{m}/total_area.npy')        
        area2 = np.load(f'{path}/{d}/{m}/total_area_squared.npy')        

        # std = np.sqrt(flux2 - flux**2)
        # std_err = std*np.sqrt(area2)/area
        # np.save(f'{path}/{d}/{m}/std.npy', std)
        # np.save(f'{path}/{d}/{m}/std_err.npy', std_err)
        
#        std_err = np.load(f'{path}/{d}/{m}/std_err.npy')
        
        # mx = np.max((mx, std_err.max()))
        
