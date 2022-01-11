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

flux = np.zeros((27, 24, 8461))
std = np.zeros((27, 24, 8461))
std_err = np.zeros((27, 24, 8461))
area = np.zeros((27, 24))


#flux = np.zeros((12, 24, 8461))
#std = np.zeros((12, 24, 8461))
#std_err = np.zeros((12, 24, 8461))
#area = np.zeros((12, 24))

used = [0, 1, 3, 4, 9, 10, 12, 13, 18, 19, 21, 22]

for d, domain in enumerate(domains[used]):
    print(domain)
    for m, month in enumerate(months):
        print(month)
        path = '/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/'\
                'test/{}/{}/'.format(domain, month)
#        flux[d, m, :] = np.load(f'{path}/avg_flux.npy', allow_pickle=False,
#                                fix_imports=False)
        area[d, m] = np.load(f'{path}/total_area.npy', allow_pickle=False,
                             fix_imports=False)
        std[d, m] = np.load(f'{path}/std.npy', allow_pickle=False,
                             fix_imports=False)
        std_err[d, m] = np.load(f'{path}/std_err.npy', allow_pickle=False,
                             fix_imports=False)

#        flux[d, m, :] = np.load(f'{path}/inferred_avg_flux.npy', allow_pickle=False,
#                     fix_imports=False)
#        area[d, m] = np.load(f'{path}/inferred_total_area.npy', allow_pickle=False,
#                       fix_imports=False)


# %%
flux_year = np.zeros((27, 4, 8461))
std_year = np.zeros((27, 4, 8461))
std_err_year = np.zeros((27, 4, 8461))
area_year = np.zeros((27, 4))

for d, domain in enumerate(domains[used]):
    for i in range(flux_year.shape[1]):
#        flux_year[d, i, :] = np.average(flux[d, 6*i:6*(i+1)],
#                                        weights=area[d, 6*i:6*(i+1)], axis=0)
#        area_year[d, i] = np.sum(area[d, 6*i:6*(i+1)])
        std_year[d, i, :] = np.average(std[d, 6*i:6*(i+1)],
                                        weights=area[d, 6*i:6*(i+1)], axis=0)
        std_err_year[d, i, :] = np.average(std_err[d, 6*i:6*(i+1)],
                                        weights=area[d, 6*i:6*(i+1)], axis=0)



# %%        
#np.save('/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/test/'
#        'monthly_mean_fluxes.npy', flux)
#np.save('/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/test/'
#        'monthly_total_area.npy', area)
np.save('/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/test/'
        'used_monthly_std.npy', std[:12])
np.save('/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/test/'
        'used_monthly_std_err.npy', std_err[:12])



# np.save('/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/test/'
#         'yearly_mean_fluxes.npy', flux_year)
# np.save('/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/test/'
#         'yearly_total_area.npy', area_year)

#np.save('/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/test/'
#        'inferred_monthly_mean_fluxes.npy', flux)
#np.save('/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/test/'
#        'inferred_monthly_total_area.npy', area)
#
#np.save('/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/test/'
#        'inferred_yearly_mean_fluxes.npy', flux_year)
#np.save('/work/um0878/user_data/froemer/rare_mistral/data/IASI/test_new/test/'
#        'inferred_yearly_total_area.npy', area_year)