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


def orbits_to_monthly_mean(domain):
    from scipy import stats
    import pandas as pd
    import glob

    path = '/work/um0878/user_data/froemer/rare_mistral/data/IASI/final/'\
           f'{domain}'

    fluxes_all = np.zeros((int(len(glob.glob(path + '/flux*', recursive=True))),
                           8461))
    area_all = np.zeros((int(len(glob.glob(path + '/area*', recursive=True)))))
    error_all = np.zeros((int(len(glob.glob(path + '/error*', recursive=True))),
                          8461))
    std_all = np.zeros((int(len(glob.glob(path + '/std*', recursive=True))),
                          8461))
    
    i, j, k, l = np.zeros(4, dtype=int)


    # Extract items containing numbers in name
    file = glob.glob(path + '/flux*', recursive=True)
    file2 = glob.glob(path + '/area*', recursive=True)
    file3 = glob.glob(path + '/error*', recursive=True)
    file4 = glob.glob(path + '/std*', recursive=True)

    # Filter only files
    file = [f for f in file if os.path.isfile(f)]
    file2 = [f for f in file2 if os.path.isfile(f)]
    file3 = [f for f in file3 if os.path.isfile(f)]
    file4 = [f for f in file4 if os.path.isfile(f)]

    for filename in file:
        fluxes_all[i, :] = np.load(filename)
        i += 1

    for filename in file2:
        area_all[j] = np.load(filename)
        j += 1

    for filename in file3:
        error_all[k, :] = np.load(filename)
        k += 1

    for filename in file4:
        std_all[l, :] = np.load(filename)
        l += 1

    # sort out outliers (more than 5 sigma) before calcuating average
    # number of standard deviations as threshold
    # also sort out nans
    n = 5
    fluxes = pd.DataFrame(fluxes_all)
    area = pd.DataFrame(area_all)
    error = pd.DataFrame(error_all)
    dev = pd.DataFrame(std_all)

    fluxes2 = fluxes[(np.abs(stats.zscore(fluxes, nan_policy='omit')) < n)
                     .all(axis=1)]
    error2 = error[(np.abs(stats.zscore(fluxes, nan_policy='omit')) < n)
                    .all(axis=1)]
    dev2 = dev[(np.abs(stats.zscore(fluxes, nan_policy='omit')) < n)
                    .all(axis=1)]
    area2 = area[(np.abs(stats.zscore(fluxes, nan_policy='omit')) < n)
                .all(axis=1)]
    
    fluxes3 = fluxes2[(np.abs(stats.zscore(dev2, nan_policy='omit')) < n)
                     .all(axis=1)]
    error3 = error2[(np.abs(stats.zscore(dev2, nan_policy='omit')) < n)
                    .all(axis=1)]
    dev3 = dev2[(np.abs(stats.zscore(dev2, nan_policy='omit')) < n)
                    .all(axis=1)]
    area3 = area2[(np.abs(stats.zscore(dev2, nan_policy='omit')) < n)
                .all(axis=1)]

    fluxes4 = fluxes3[(np.abs(stats.zscore(error3, nan_policy='omit')) < n)
                     .all(axis=1)]
    error4 = error3[(np.abs(stats.zscore(error3, nan_policy='omit')) < n)
                    .all(axis=1)]
    dev4 = dev3[(np.abs(stats.zscore(error3, nan_policy='omit')) < n)
                    .all(axis=1)]
    area4 = area3[(np.abs(stats.zscore(error3, nan_policy='omit')) < n)
                .all(axis=1)]

    fluxes = np.array(fluxes4)
    error = np.array(error4)
    dev = np.array(dev4)
    area = np.array(area4)[:, 0]

    squared = fluxes**2
    avg_squared = np.average(squared, weights=area, axis=0)

    squared_error = error**2
    avg_squared_error = np.average(squared_error, weights=area, axis=0)

    squared_std = dev**2
    avg_squared_std = np.average(squared_std, weights=area, axis=0)

    avg = np.average(fluxes, weights=area, axis=0)
    # std = np.std(fluxes, axis=0)

    avg_error = np.average(error, weights=area, axis=0)
    avg_std = np.average(dev, weights=area, axis=0)

    total_area = np.sum(area)
    total_area_squared = np.sum(area**2)

    np.save(f'{path}/avg_flux.npy', avg, allow_pickle=False, fix_imports=False)
    np.save(f'{path}/avg_flux_squared.npy', avg_squared, allow_pickle=False,
            fix_imports=False)
    np.save(f'{path}/total_area.npy', total_area,
            allow_pickle=False, fix_imports=False)
    np.save(f'{path}/total_area_squared.npy', total_area_squared,
            allow_pickle=False, fix_imports=False)

    np.save(f'{path}/avg_error.npy', avg_error,
            allow_pickle=False, fix_imports=False)
    np.save(f'{path}/avg_error_squared.npy', avg_squared_error,
            allow_pickle=False, fix_imports=False)
    np.save(f'{path}/avg_std.npy', avg_std,
            allow_pickle=False, fix_imports=False)
    np.save(f'{path}/avg_std_squared.npy', avg_squared_std,
            allow_pickle=False, fix_imports=False)

    return fluxes, avg, area


if __name__ == '__main__':
    start = time.process_time()
    # os.chdir('/work/um0878/user_data/froemer/rare_mistral/scripts/')

#    year = sys.argv[1]
#    month = sys.argv[2]

    # year, month = '2011', '01'

#    print(year, month)

    used = [0, 1, 3, 4, 9, 10, 12, 13, 18, 19, 21, 22]

    domains = np.load('/work/um0878/user_data/froemer/rare_mistral/data/IASI/'
                      'test_new/test/domains.npy')[used]

#    months = []
#    for year in [2011, 2013, 2016, 2017]:
#        for month in ['01', '03', '05', '07', '09', '11']:
#            months.append(f'{year}/{month}')
#    np.save('/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/'
#            'test/months.npy', months)

    for d, domain in enumerate(domains):
        print(domain)
        flux, avg, area = orbits_to_monthly_mean(domain)

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
    # length = np.min(np.array((len(fluxes[0]), len(fluxes[1]),
    #                 len(fluxes[2]))))
    # extra = (fluxes[0][:length]*areas[0][:length].reshape(
    #                   areas[0][:length].shape[0], 1) -
    #           fluxes[1][:length]*areas[1][:length].reshape(
    #                   areas[1][:length].shape[0], 1))/ \
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

    # avg_est = (average[0]*total_area[0] - average[1]*total_area[1]) / \
    #                   total_area[2]

    # path = '/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/'\
    #        'test/{}/{}/{}/'.format(domain, year, month)

    # np.save(f'{path}/inferred_avg_flux.npy', avg_est, allow_pickle=False,
    #         fix_imports=False)
    # np.save(f'{path}/inferred_total_area.npy', total_area[2],
    #         allow_pickle=False, fix_imports=False)
