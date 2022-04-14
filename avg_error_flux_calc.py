#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 09:10:51 2021

@author: u301023
"""

import matplotlib.pyplot as plt
import numpy as np


import functions as f

path1 = '/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/test'
path2 = '/work/um0878/user_data/froemer/rare_mistral/data/IASI/final'

used = [0, 1, 3, 4, 9, 10, 12, 13, 18, 19, 21, 22]
domains = np.load(f'{path1}/domains.npy')[used]
chan = np.load(f'/work/um0878/user_data/froemer/rare_mistral/data/chan',
               allow_pickle=True)[-8461:]

dchan = 0.25
error = np.zeros((len(domains), 8461))
flux = np.zeros((len(domains), 8461))

color = ['k', 'r', 'b']
lines = ['-', ':']
alpha = [1, 0.5]

plt.close('all')
plt.figure(1, figsize=(12, 6))
plt.xlabel('wavenumber (cm-1)', fontsize=18)
plt.ylabel('mean error (mW/m2/cm-1)', fontsize=18)

plt.figure(2, figsize=(12, 6))
plt.xlabel('wavenumber (cm-1)', fontsize=18)
plt.ylabel('mean error (%)', fontsize=18)

for d, dom in enumerate(domains):
    # error[d] = np.load(f'{path2}/{dom}/avg_error.npy')
    error[d] = np.load(f'{path2}/{dom}/avg_std.npy')
    flux[d] = np.load(f'{path2}/{dom}/avg_flux.npy')
    plt.figure(1)
    plt.plot(chan, f.mov_avg(1000*error[d], 200), color=color[int(d/4)],
             linestyle=lines[int(d/2)%2], alpha=alpha[d%2],
             label=f'{dom}: {np.round(np.sqrt(np.sum((dchan*error[d]/2)**2)), 3)} W/m2')
    #         label=f'{dom}: {np.round(np.sum((dchan*error[d]/2)), 2)}')
    # if we assume errors to be random, sqrt(sum(squares)) is correct
    # if errors are not random, upper limit of error is sum()
    # this must be consistent for both spectral AND angular integral!!!

    plt.figure(2)
    plt.plot(chan, f.mov_avg(100*error[d]/flux[d], 200), color=color[int(d/4)],
             linestyle=lines[int(d/2)%2], alpha=alpha[d%2],
             label=f'{dom}: {np.round(100*error[d].mean()/flux[d].mean(), 2)}%')

    
plt.figure(1)
plt.legend(fontsize=14)

plt.figure(2)
plt.legend(fontsize=14)

plt.show()