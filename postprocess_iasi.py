#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 12:04:49 2021

@author: u301023
"""
import sys
import os
import time
import numpy as np


def orbits_to_monthly_mean(domain, year, month):
    from scipy import stats
    import pandas as pd
    import glob

    path = '/scratch/uni/u237/user_data/vojohn/iasi_flux/'\
            f'{domain}/{year}/{month}'
    path2 = '/scratch/uni/u237/user_data/froemer/iasi/data/'\
            f'{domain}/{year}/{month}'
    
    fluxes_all = np.zeros((int(len(glob.glob(f'{path}/flux*', recursive=True))),
                               8461))
    area_all = np.zeros((int(len(glob.glob(f'{path}/area*', recursive=True)))))
    i, j = np.zeros(2, dtype=int)

    # Extract items containing numbers in name
    file = glob.glob(f'{path}/flux*', recursive=True)
    file2 = glob.glob(f'{path}/area*', recursive=True)

    # Filter only files
    file = [f for f in file if os.path.isfile(f)]
    file2 = [f for f in file2 if os.path.isfile(f)]

    for filename in file:
        fluxes_all[i, :] = np.load(filename)
        i += 1

    for filename in file2:
        area_all[j] = np.load(filename)
        j += 1

    # sort out outliers (more than 5 sigma) before calcuating average
    # number of standard deviations as threshold
    # also sort out nans
    n = 5
    fluxes = pd.DataFrame(fluxes_all)
    area = pd.DataFrame(area_all)
    fluxes2 = fluxes[(np.abs(stats.zscore(fluxes, nan_policy='omit')) < n)
                     .all(axis=1)]
    area = area[(np.abs(stats.zscore(fluxes, nan_policy='omit')) < n)
                .all(axis=1)]

    fluxes = np.array(fluxes2)
    area = np.array(area)[:, 0]

    squared = fluxes**2
    avg_squared = np.average(squared, weights=area, axis=0)

    avg = np.average(fluxes, weights=area, axis=0)
    std = np.std(fluxes, axis=0)

    total_area = np.sum(area)

    np.save(f'{path2}/avg_flux.npy', avg, allow_pickle=False, fix_imports=False)
    np.save(f'{path2}/avg_flux_squared.npy', avg_squared, allow_pickle=False,
            fix_imports=False)
    np.save(f'{path2}/total_area.npy', total_area,
            allow_pickle=False, fix_imports=False)

    return fluxes, avg, std, area


if __name__ == '__main__':
    start = time.process_time()
    os.chdir('/scratch/uni/u237/user_data/froemer/iasi/')

#    year = sys.argv[1]
#    month = sys.argv[2]
    
    year, month = '2007', '07'

    print(year, month)

    domains = ['global/all-sky/land+ocean',
              'global/all-sky/ocean-only',
              'global/clear-sky/land+ocean',
              'global/clear-sky/ocean-only',
              'tropics/all-sky/land+ocean',
              'tropics/all-sky/ocean-only',
              'tropics/clear-sky/land+ocean',
              'tropics/clear-sky/ocean-only',
              'extra/all-sky/land+ocean',
              'extra/all-sky/ocean-only',
              'extra/clear-sky/land+ocean',
              'extra/clear-sky/ocean-only']
    
#
#    np.save('/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/'
#            'test/domains.npy', domains)
#
#    months = []
#    for year in [2011, 2013, 2016, 2017]:
#        for month in ['01', '03', '05', '07', '09', '11']:
#            months.append(f'{year}/{month}')
#    np.save('/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/'
#            'test/months.npy', months)

    for d, domain in enumerate(domains):
        print(domain)
        flux, avg, std, area = orbits_to_monthly_mean(
            domain, year, month)

    end = time.process_time()
    print(end - start)
    
    # %% inferring orbit wise (option 1) does deliver worse results if number 
    # of orbits is different between domains: when averaging, the wrong weights
    # are assigned from the first time one orbit is nan onwards
    # mean value is better than that, but still worse compared to average with
    # correctly assinged weights
    # solution: calculate monthly average separately for every domains with 
    # correct weights, then infer from those monthly means using monthly total
    # areas
    
    # option 1
    # length = np.min(np.array((len(fluxes[0]), len(fluxes[1]), len(fluxes[2]))))
    # extra = (fluxes[0][:length]*areas[0][:length].reshape(areas[0][:length].shape[0], 1) - 
    #           fluxes[1][:length]*areas[1][:length].reshape(areas[1][:length].shape[0], 1))/ \
    #         areas[2][:length].reshape(areas[2][:length].shape[0], 1)
            
    # extra_area = areas[0] - areas[1]

    # option 2
    # domains = ['global/clear-sky/land+ocean',
    #        'tropics/clear-sky/land+ocean',
    #        'extra/clear-sky/land+ocean']

    # fluxes = []
    # areas = []
    # average = np.zeros((3, 8461))
    # total_area = np.zeros((3))
    # for d, domain in enumerate(domains):
    #     print(domain)
    #     flux, avg, std, area = orbits_to_monthly_mean(
    #         domain, year, month)
    #     fluxes.append(flux)
    #     areas.append(area)
    #     average[d, :] = np.average(flux.T, weights=area.flatten(), axis=1)
    #     total_area[d] = np.sum(area)
        
    # avg_est = (average[0]*total_area[0] - average[1]*total_area[1]) / total_area[2]

    # path = '/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/'\
    #        'test/{}/{}/{}/'.format(domain, year, month)

    # np.save(f'{path}/inferred_avg_flux.npy', avg_est, allow_pickle=False,
    #         fix_imports=False)
    # np.save(f'{path}/inferred_total_area.npy', total_area[2],
    #         allow_pickle=False, fix_imports=False)
