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


def masked_average(radiance, angle, mask, lat, scale):
    '''
    This functions averages the fluxes along track as well as over each
    scanning angle (4 pixels per scanning angle), applying the three different
    masks.
    '''
    weight_sum = np.zeros((30, 1))
    weight_sum = np.sum(mask * scale, axis=(0, 2))

    # apply masks and average over all scans and all pixels per scan
    rad = np.divide(np.sum(radiance * mask, axis=(0, 2)),
                    weight_sum)
    ang = np.divide(np.sum(angle * (mask * scale)[:, :, :, 0], axis=(0, 2)),
                    weight_sum[:, 0])

    return rad, ang, weight_sum


def average_symmetric_angles(ang, rad, weight_sum):
    '''
    Assuming azimuthal symmetry, corresponding (opposite) scanning angles are
    averaged together.
    '''
    # average over symmetric zenith angles (apply weights from before)
    weight_sum = weight_sum[:, 0]
    meanangle = np.zeros((15))
    meanrad = np.zeros((15, 8461))
    for m in range(len(meanangle)):
        meanangle[-m-1] = np.average((ang[m], ang[-m-1]), axis=0,
                                     weights=(weight_sum[m], weight_sum[-m-1]))
        meanrad[-m-1] = np.average(np.array((rad[m, :], rad[-m-1, :])), axis=0,
                                   weights=(weight_sum[m], weight_sum[-m-1]))

    return meanangle, meanrad


def prepare_interpolation(meanangle, meanrad):
    '''
    Initialises arrays for angular interpolation (weighted with cosine of
    zenith angle).
    '''
    # weight radiance with cosine of zenith angle
    cosangle = np.zeros((15, 1))
    cosangle[:, 0] = np.cos(np.deg2rad(meanangle))
    radcos = np.zeros((meanrad.shape[0]+7, 8461))
    radcos[:15, :] = meanrad*cosangle

    # extend zenith angles with similar angular resolution
    fullangle = np.zeros((radcos.shape[0], 1))
    fullangle[:15, 0] = meanangle
    fullangle[15:, 0] = np.linspace(fullangle[14, 0], 90, 8)[1:]

    return fullangle, radcos


def interpolate(fullangle, radcos):
    '''
    Performs linear angular interpolations of spectral I*cos(theta).
    '''
    for i in range(radcos.shape[1]):
        interp = scipy.interpolate.interp1d((fullangle[14, 0], 90),
                                            (radcos[14, i], 0), 'linear')
        radcos[15:, i] = interp(fullangle[15:, 0])

    return radcos


def calc_specflux(radcos, fullangle):
    '''
    Integrates over all wavenumber to calculated spectral flux.
    '''
    specflux = np.zeros(radcos.shape[1])
    for i in range(specflux.shape[0]):
        specflux[i] = scipy.integrate.trapz(
            radcos[:, i]*np.sin(np.deg2rad(fullangle[:, 0])),
            np.deg2rad(fullangle[:, 0]))*2*np.pi

    return specflux


def save_flux(specflux, area, year, month, orbit, domain):
    '''
    Average spectral fluxes and total area are saved.
    '''
    path = '/work/um0878/user_data/froemer/rare_mistral/data/IASI/final/'\
           f'{domain}/{year}/{month}'
    np.save(f'{path}/area_{orbit}', area,
            allow_pickle=False, fix_imports=False)
    np.save(f'{path}/flux_{orbit}', specflux,
            allow_pickle=False, fix_imports=False)


def process_data(radiance, angle, mask, domain, orbit, lat, scale):
    '''
    Encapsulates processing steps applied to the different domains.
    '''
    area = np.sum(mask*scale)
    rad, ang, weight_sum = masked_average(radiance, angle, mask, lat, scale)
    meanangle, meanrad = average_symmetric_angles(ang, rad, weight_sum)
    fullangle, radcos = prepare_interpolation(meanangle, meanrad)
    radcos = interpolate(fullangle, radcos)
    specflux = calc_specflux(radcos, fullangle)
    save_flux(specflux, area, year, month, orbit, domain)

    print(f'output flux for domain {domain}: {specflux[1000]}')


def main(FILE):
    # read IASI data
    idata = read_data(FILE)

    # assign
    orbit = FILE[74:105]
    radiance = idata.spectra_orbit * 100  # convert units to W cm m-2 sr-1
    lat = idata.lat_orbit
    angle = idata.GGeoSondAnglesMETOP[:, :, :, 0]
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

    for dom1 in ['global', 'tropics', 'extra']:
        for dom2 in ['all-sky', 'clear-sky']:
            for dom3 in ['land+ocean', 'ocean-only']:
                domain = f'{dom1}/{dom2}/{dom3}'
                mask = create_mask(lat, land_frac, cloud_frac, domain)
                process_data(radiance, angle, mask, domain, orbit, lat, scale)


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
