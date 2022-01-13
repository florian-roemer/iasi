#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 09:15:26 2021

@author: u301023
"""

import numpy as np


domains = np.load('/work/um0878/user_data/froemer/rare_mistral/data/IASI/'
                  'test_new/test/domains.npy')
#domains = ['extra/clear-sky/land+ocean']

months = np.load('/work/um0878/user_data/froemer/rare_mistral/data/IASI/'
                  'test_new/test/months.npy')

std = np.zeros((12, 8461)) # standard deviation assumed constant for each month
std_err = np.zeros((12, 24, 8461))

used = [0, 1, 3, 4, 9, 10, 12, 13, 18, 19, 21, 22]

for d, domain in enumerate(domains[used]):
    print(domain)
    path = '/work/um0878/user_data/froemer/rare_mistral/data/IASI/'\
            f'final/{domain}/'
    std[d] = np.load(f'{path}/avg_std.npy',
                     allow_pickle=False, fix_imports=False)

    for m, month in enumerate(months):
        print(month)
        path = '/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/'\
                f'test/{domain}/{month}'
        std_err[d, m] = np.load(f'{path}/std_err_new.npy',
                                allow_pickle=False, fix_imports=False)[:, 0]


np.save('/work/um0878/user_data/froemer/spectral_feedbacks/IASI/'
        'new_used_avg_std.npy', std)
np.save('/work/um0878/user_data/froemer/spectral_feedbacks/IASI/'
        'new_used_monthly_std_err.npy', std_err)


