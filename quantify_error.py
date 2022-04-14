#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:18:50 2021

@author: u301023
"""

import numpy as np
import glob
import sys

year = sys.argv[1]
month = sys.argv[2]

#year, month = '2017', '01'

#PATH = f'/work/um0878/data/iasi/iasi-l1/reprocessed/m02/'\
#       f'{year}/{month}/{day}/'

#orbitlist = []
#
#for FILE in np.sort(glob.glob(PATH + 'IASI*', recursive=False))[:14]:
#    orbitlist.append(FILE[74:105])

path = '/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/test/'
domain = 'extra/cloudy/land-only'
n = len(np.sort(glob.glob(path + domain + f'/{year}/{month}/area*',
                          recursive=False)))

name = []
area = np.zeros((27, n, 1))
flux = np.zeros((27, n, 8461))
i, j = 0, 0
for dom1 in ['global', 'tropics', 'extra']:
    for dom2 in ['all-sky', 'clear-sky', 'cloudy']:
        for dom3 in ['land+ocean', 'ocean-only', 'land-only']:
            domain = f'{dom1}/{dom2}/{dom3}'
            print(domain)
            name.append(domain)
            print(len(np.sort(glob.glob(
                    path + domain + f'/{year}/{month}/area*',
                    recursive=False))))
            for i, FILE in enumerate(np.sort(glob.glob(
                    path + domain + f'/{year}/{month}/area*',
                    recursive=False))[:n]):
                orbit = FILE[-35:-4]
                area[j, i, 0] = np.load(
                        f'{path}/{domain}/{year}/{month}/area_{orbit}.npy')
                flux[j, i, :] = np.load(
                        f'{path}/{domain}/{year}/{month}/flux_{orbit}.npy')

                i += 1
            j += 1

home = '/mnt/lustre01/pf/zmaw/u301023/iasi'
np.save(f'{home}/stuff/data/{year}/flux_{year}_{month}.npy', flux)
np.save(f'{home}/stuff/data/{year}/area_{year}_{month}.npy', area)
