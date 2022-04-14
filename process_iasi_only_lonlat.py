#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This scripts filters and averages IASI spectral radiances from one orbit and
calculates spectral fluxes using angular interpolation and integration.
It currently calculates the mean spectral fluxes over 12 domains, namely
all combinations of:
    1) global, tropics and extra-tropics
    2) all-sky and clear-sky
    3) land+ocean and ocean-only

The script takes three input parameters:
    1) the year,
    2) the month,
    3) and the day

that should be processed. The script then loops over all orbits on that day
and calculates average spectral fluxes for each of them separately.
The output are two .npy type files. They can be read using numpy's
numpy.load routine (pickle=False):
    1) flux_<DATE_AND_TIME_OF_ORBIT>: spectral flux for each of IASI's 8461
    channels
    2) area_<DATE_AND_TIME_OF_ORBIT>: total area (number of pixels weighted by
    the cosine of the latitude and corrected for oversampling)

They are needed for calculating monthly weighted averages of multiple orbits.

To run this script on mistral first run in terminal:
module load anaconda3/.bleeding_edge

@author: Florian Roemer (florian.roemer@uni-hamburg.de)
"""

import glob
import numpy as np
import scipy.integrate
import scipy.interpolate
import sys
import time

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


def create_mask(lat, land_frac, cloud_frac, domain):
    '''
    This function creates the mask for the selected domain.
    '''
    # initialize
    mask = np.full((lat.shape[0], lat.shape[1], lat.shape[2], 1),
                   True, dtype=bool)

    # filters
    trop = (abs(lat) < 30)
    clear_sky = (cloud_frac == 0)
    ocean = (land_frac == 0)

    # create masks
    if 'tropics' in domain:
        mask[:, :, :, 0] = np.logical_and(mask[:, :, :, 0], trop[:, :, :, 0])
    if 'extra' in domain:
        mask[:, :, :, 0] = np.logical_and(mask[:, :, :, 0], ~trop[:, :, :, 0])
    if 'clear-sky' in domain:
        mask[:, :, :, 0] = np.logical_and(mask[:, :, :, 0], clear_sky)
    if 'ocean-only' in domain:
        mask[:, :, :, 0] = np.logical_and(mask[:, :, :, 0], ocean)

    return mask


def save(year, month, orbit, lat, lon, cloud_frac):
    '''
    Average spectral fluxes and total area are saved.
    '''
    path = '/work/um0878/user_data/froemer/rare_mistral/data/IASI/final/lonlat'
    np.save(f'{path}/lat_{orbit}', lat,
            allow_pickle=False, fix_imports=False)
    np.save(f'{path}/lon_{orbit}', lon,
            allow_pickle=False, fix_imports=False)
    np.save(f'{path}/cf_{orbit}', cloud_frac,
            allow_pickle=False, fix_imports=False)


def main(FILE):
    # read IASI data
    idata = read_data(FILE)

    # assign
    orbit = FILE[74:105]
    radiance = idata.spectra_orbit * 100  # convert units to W cm m-2 sr-1
    lat = idata.lat_orbit
    lon = idata.lon_orbit
    cloud_frac = idata.GEUMAvhrr1BCldFrac[:, :, :]
    land_frac = idata.GEUMAvhrr1BLandFrac[:, :, :]

    # ensure consistent dimensions
    lat = lat.reshape(lat.shape[0], 30, 4, 1)
    radiance = radiance.transpose((0, 2, 3, 1))

    # read factor correcting systematic oversampling of sub-polar latitudes
    # using 180 bins from -90 to 90, they are centered at -89.5, -88.5 ... 89.5
    factor = np.load('/pf/u/u301023/iasi/factor.npy')
    index = np.round(lat + 89.5).astype('int')
    weight = factor[index]
    cos = np.cos(np.deg2rad(lat))
    scale = cos*weight

    # scale radiance with cosine of latitude and correct oversampling of
    # sup-polar latitudes
    radiance = radiance * scale
    
    save(year, month, orbit, lat, lon, cloud_frac)


if __name__ == '__main__':
    start = time.process_time()

    year = sys.argv[1]
    month = sys.argv[2]
    day = sys.argv[3]

    PATH = f'/work/um0878/data/iasi/iasi-l1/reprocessed/m02/'\
           f'{year}/{month}/{day}/'

    for FILE in np.sort(glob.glob(PATH + 'IASI*', recursive=False)):
        print(f'Processing orbit {FILE[74:105]}')
        try:
            main(FILE)
        except:
            print(f'Invalid value encountered in {FILE}.'
                  'Skipping this orbit.')
            pass

    end = time.process_time()
    print(f'Your program needed {(end-start)/60} minutes.')
