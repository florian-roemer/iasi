#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 12:04:49 2021

@author: u301023
"""
import sys
import os
import time


def orbits_to_monthly_mean(domain, year, month):
    import numpy as np
    from scipy import stats
    import pandas as pd

    import glob
    import os

    path = '../data/IASI/test_new/{}/{}/{}/'.format(domain, year, month)

    fluxes_all = np.zeros((int(len(os.listdir(path))/2), 8461))
    nobs_all = np.zeros((int(len(os.listdir(path))/2)))
    i, j = np.zeros(2, dtype=int)

    # Extract items containing numbers in name
    file = glob.glob(path + 'flux*', recursive=True)
    file2 = glob.glob(path + 'nobs*', recursive=True)

    # Filter only files
    file = [f for f in file if os.path.isfile(f)]
    file2 = [f for f in file2 if os.path.isfile(f)]

    for filename in file:
        fluxes_all[i, :] = np.load(filename)
        i += 1

    for filename in file2:
        nobs_all[j] = np.load(filename)
        j += 1

    # sort out outliers (more than 3 sigma) before calcuating average
    # number of standard deviations as threshold
    n = 3
    fluxes = pd.DataFrame(fluxes_all)
    nobs = pd.DataFrame(nobs_all)
    fluxes2 = fluxes[(np.abs(stats.zscore(fluxes)) < n).all(axis=1)]
    nobs = nobs[(np.abs(stats.zscore(fluxes)) < n).all(axis=1)]

    fluxes = np.array(fluxes2)
    nobs = np.array(nobs)[:, 0]

    squared = fluxes**2
    avg_squared = np.average(squared, weights=nobs, axis=0)

    avg = np.average(fluxes, weights=nobs, axis=0)
    std = np.std(fluxes, axis=0)

    np.save(f'{path}/mean_flux', avg, allow_pickle=False, fix_imports=False)
    np.save(f'{path}/mean_flux_squared', avg_squared, allow_pickle=False,
            fix_imports=False)
    np.save(f'{path}/total_orbits', fluxes.shape[0],
            allow_pickle=False, fix_imports=False)

    return fluxes, avg, std


if __name__ == '__main__':
    start = time.process_time()
    os.chdir('/work/um0878/user_data/froemer/rare_mistral/scripts/')

    domain = sys.argv[1]
    year = sys.argv[2]
    month = sys.argv[3]

    fluxes, avg, std = orbits_to_monthly_mean(domain, year, month)

    end = time.process_time()
    print(end - start)
