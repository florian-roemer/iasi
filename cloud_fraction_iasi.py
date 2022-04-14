#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:13:13 2021

@author: u301023
"""

import time
import sys
import glob

import numpy as np
import read_iasi as ri


def read_data(FILEPATH):
    '''
    read IASI data from eps file using readin_eps reading routine
    '''
    idata = ri.data(FILEPATH)
    idata.get_orbit_lat_lon()
    idata.get_spec_orbit()
    print(f'input radiance: {idata.spectra_orbit[0, 1000:1005, 0, 0]}')

    return idata


def main(FILE):
    # read IASI data
    idata = read_data(FILE)

    # assign
    orbit = FILE[74:105]

    lat = idata.lat_orbit
    lat = lat.reshape(lat.shape[0], 30, 4, 1)

    lon = idata.lon_orbit
    lon = lon.reshape(lon.shape[0], 30, 4, 1)

    cloud_frac = idata.GEUMAvhrr1BCldFrac[:, :, :]

    np.save('/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/'
            f'test/cloud_fraction_test/lat_{orbit}.npy', lat)
    np.save('/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/'
            f'test/cloud_fraction_test/lon_{orbit}.npy', lon)
    np.save('/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/'
            f'test/cloud_fraction_test/cloud_frac_{orbit}.npy', cloud_frac)
    


if __name__ == '__main__':
    start = time.process_time()

#    year = sys.argv[1]
#    month = sys.argv[2]
#    day = sys.argv[3]

    year, month, day = '2017', '11', '30'

    PATH = f'/work/um0878/data/iasi/iasi-l1/reprocessed/m02/'\
           f'{year}/{month}/{day}/'

    for FILE in np.sort(glob.glob(PATH + 'IASI*', recursive=False)):
        print(f'Processing orbit {FILE[74:105]}')
        main(FILE)

    end = time.process_time()
    print(f'Your program needed {(end-start)/60} minutes.')
