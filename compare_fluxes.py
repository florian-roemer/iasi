#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 09:19:51 2021

@author: u301023
"""

import numpy as np
import matplotlib.pyplot as plt

def planck_wavenumber(n, T):
    """Calculate black body radiation for given wavenumber and temperature.

    Parameters:
        n (float or ndarray): Wavenumber.
        T (float or ndarray): Temperature [K].

    Returns:
        float or ndarray: Radiances.

    """
    c = 2.99792458e8
    h = 6.62607015e-34
    k = 1.380649e-23

    return 2 * h * c**2 * n**3 / (np.exp(np.divide(h * c * n, (k * T))) - 1)


chan = np.load('/work/um0878/user_data/froemer/rare_mistral/data/chan',
               allow_pickle=True)

planck_tropics = planck_wavenumber(chan*100, 300) * 100 * 2.87
planck_global = planck_wavenumber(chan*100, 288) * 100 * 2.87

flux = np.zeros((8, 8461))
i = 0

plt.close('all')

plt.figure(1)
plt.title('tropics')
plt.plot(chan, planck_tropics, color='k', label='Planck')

plt.figure(2)
plt.title('global')
plt.plot(chan, planck_global, color='k', label='Planck')


for dom1 in ['global', 'tropics']:
    for dom2 in ['all-sky', 'clear-sky']:
        for dom3 in ['land+ocean', 'ocean-only']:
            domain = f'{dom1}/{dom2}/{dom3}'
            flux[i, :] = np.load('/work/um0878/user_data/froemer/rare_mistral/'
                                 f'data/IASI/test_new/test/{domain}/2011/01/'
                                 'flux_20110101003556Z_20110101021755Z.npy')
            if 'tropics' in domain:
                plt.figure(1)
            else:
                plt.figure(2)
            plt.plot(chan[-8461:], flux[i, :], label=domain)
            i += 1
            
plt.figure(1)
plt.legend()
plt.figure(2)
plt.legend()

