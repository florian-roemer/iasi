#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:29:05 2021

@author: u301023
"""

import numpy as np
import matplotlib.pyplot as plt
import glob

chan = np.load('/work/um0878/user_data/froemer/rare_mistral/data/chan',
               allow_pickle=True)

path = '/work/um0878/user_data/froemer/rare_mistral/data/IASI'

used = [0, 1, 3, 4, 9, 10, 12, 13, 18, 19, 21, 22]
domains = np.load(f'{path}/test_new/test/domains.npy')[used]
months = np.load(f'{path}/test_new/test/months.npy')
mx = 0
# plt.figure()
for d in domains:
    print(d)
    for m in months:
        total_area = np.load(f'{path}/test_new/test/{d}/{m}/total_area.npy')        
        files = glob.glob(f'{path}/test_new/test/{d}/{m}/area*', recursive=True)
        
        std_err = np.load(f'{path}/final/{d}/avg_error.npy')
        std_err = std_err.reshape((std_err.size, 1))


        area = np.zeros((len(files)))
        for f, file in enumerate(files):
            area[f] = np.load(file)
            
        area_frac = area/total_area
        area_frac = area_frac.reshape((1, area_frac.size))
        
        final_std_err = np.sqrt(np.sum((area_frac*std_err)**2, axis=1))
        
        # mean_std = np.sqrt(np.sum((area-1)*std**2, axis=1)/ (np.sum(area-1)))

        # area_frac2 = np.full(area.shape, 1/413)
        # final_std2 = np.sqrt(np.sum((std*area_frac2)**2, axis=1))

        # std = np.sqrt(flux2 - flux**2)
        # std_err = std*np.sqrt(area2)/area
        # np.save(f'{path}/{d}/{m}/std.npy', std)
        np.save(f'{path}/test_new/test/{d}/{m}/std_err_new.npy', std_err)
        
#        std_err = np.load(f'{path}/{d}/{m}/std_err.npy')
        
        # mx = np.max((mx, std_err.max()))
        
