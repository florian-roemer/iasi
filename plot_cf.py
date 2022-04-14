#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:21:35 2021

@author: u301023
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import glob
import sys
import netCDF4 as nc
from scipy.interpolate import LinearNDInterpolator



def main(year, month, day, source):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='grey')
    ax.add_feature(cfeature.OCEAN, facecolor='grey')


    if 'ERA' in source:
        era = nc.Dataset(
            '/mnt/lustre02/work/um0878/user_data/froemer/rare_mistral/data/'
            f'ERA5/mistral/domains/cf/test/cf_{year}_{month}_{day}.nc')

        ecf = era.variables['TCC'][0]
        elat = era.variables['lat'][:]
        elon = era.variables['lon'][:]

        elon = elon.reshape((elon.size, 1))
        elon2 = np.tile(elon, elat.size)

        elat = elat.reshape((elat.size, 1))
        elat2 = np.tile(elat, elon.size)

        elon = np.transpose(elon2, (1, 0))
        elat = elat2

        cf = ax.contourf(elon, elat, ecf, transform=ccrs.PlateCarree(),
                         cmap='Blues_r')
        fig.colorbar(cf)
        ax.set_title(f'ERA5: {day}/{month}/{year}')
        fig.savefig(
            '/mnt/lustre02/work/um0878/user_data/froemer/rare_mistral/data/'
            f'ERA5/mistral/domains/cf/test/cf_map2_{year}_{month}_{day}.png',
            dpi=300)

    elif source == 'IASI':
        icf, ilat, ilon = [], [], []
        PATH = f'/work/um0878/data/iasi/iasi-l1/reprocessed/m02/'\
               f'{year}/{month}/{day}/'

        for FILE in np.sort(glob.glob(PATH + 'IASI*', recursive=False)):
            orbit = FILE[74:105]
            icf.append(np.load(
                    '/work/um0878/user_data/froemer/rare_mistral/data/'
                    'IASI/test_new/test/cloud_fraction_test/'
                    f'cloud_frac_{orbit}.npy'))
            ilat.append(np.load(
                    '/work/um0878/user_data/froemer/rare_mistral/data/'
                    'IASI/test_new/test/cloud_fraction_test/'
                    f'lat_{orbit}.npy'))
            ilon.append(np.load(
                    '/work/um0878/user_data/froemer/rare_mistral/data/'
                    'IASI/test_new/test/cloud_fraction_test/'
                    f'lon_{orbit}.npy'))

        icf = np.concatenate(icf, axis=0).flatten()/100
        ilat = np.concatenate(ilat, axis=0).flatten()
        ilon = np.concatenate(ilon, axis=0).flatten()

        x = ilon
        y = ilat
        z = icf

        X = np.linspace(min(x), max(x), 360)
        Y = np.linspace(min(y), max(y), 360)
        X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
        interp = LinearNDInterpolator(list(zip(x, y)), z)
        Z = interp(X, Y)
        cf = ax.contourf(X, Y, Z, cmap='Blues_r', transform=ccrs.PlateCarree())

#        tcf = ax.tricontourf(ilon, ilat, icf,
#                             transform=ccrs.PlateCarree(), cmap='Blues_r')
        ax.set_title(f'IASI: {day}/{month}/{year}')
        fig.colorbar(cf)

        fig.savefig(
            '/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/'
            f'test/cloud_fraction_test/cf_map_interp_{year}_{month}_{day}.png',
            dpi=300)

    else:
        print('No valid source!')


if __name__ == '__main__':
#    year, month, day = '2017', '11', '30'
    year, month, day = sys.argv[1], sys.argv[2], sys.argv[3]
#    source = sys.argv[1]
    source = 'IASI'

    main(year, month, day, source)

