#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 11:08:58 2021

@author: u301023
"""

import matplotlib.pyplot as plt
import numpy as np

#year, month = '2011', '11'

chan = np.load('/work/um0878/user_data/froemer/rare_mistral/data/chan',
               allow_pickle=True)[-8461:]

home = '/mnt/lustre01/pf/zmaw/u301023/iasi'

extra_diff_mean_month = np.zeros((24, 8461))
extra_diff_mean_year = np.zeros((4, 8461))

i = 0
for year in ['2011', '2013', '2016', '2017']:
    for month in ['01', '03', '05', '07', '09', '11']:
        print(f'{year}/{month}')
        flux = np.load(f'{home}/stuff/data/{year}/flux_{year}_{month}.npy')
        area = np.load(f'{home}/stuff/data/{year}/area_{year}_{month}.npy')
        n = flux.shape[1]
        
        name = []
        for dom1 in ['global', 'tropics', 'extra']:
            for dom2 in ['all-sky', 'clear-sky', 'cloudy']:
                for dom3 in ['land+ocean', 'ocean-only', 'land-only']:
                    domain = f'{dom1}/{dom2}/{dom3}'
                    name.append(domain)
                    
        global_area = np.zeros((1, n, 1))
        global_area[0, :, 0] = area[0, :, 0]

        # extra cloudy land, 
        # index = 26
        # avg = np.average(flux, weights=global_area.flatten(), axis=1)
        # avg_area = np.average(area, weights=global_area.flatten(), axis=1)
        # avg_est = (((avg[0]*avg_area[0] - avg[1]*avg_area[1]) -
        #              (avg[3]*avg_area[3] - avg[4]*avg_area[4])) -
        #            ((avg[9]*avg_area[9] - avg[10]*avg_area[10]) -
        #             (avg[12]*avg_area[12] - avg[13]*avg_area[13]))) / avg_area[index]
        
        # est = (((flux[0]*area[0] - flux[1]*area[1]) -
        #         (flux[3]*area[3] - flux[4]*area[4])) -
        #       ((flux[9]*area[9] - flux[10]*area[10]) -
        #         (flux[12]*area[12] - flux[13]*area[13]))) / area[index]
        
        # extra clear land+ocean 
        index = 21
        avg = np.average(flux, weights=global_area.flatten(), axis=1)
        avg_area = np.average(area, weights=global_area.flatten(), axis=1)
        avg_est = (avg[3]*avg_area[3] - avg[12]*avg_area[12]) / avg_area[index]
        
        est = (flux[3]*area[3] - flux[12]*area[12]) / area[index]
               
        extra_diff = (est - flux[index])/flux[index]
        #extra_diff_mean = np.average(extra_diff, weights=area[index].flatten(), axis=0)
        extra_diff_mean = np.nanmean(extra_diff, axis=0)
        extra_diff_mean_month[i] = extra_diff_mean

        avg_diff = (avg_est - avg[index])/avg[index]

#        plt.plot(chan, extra_diff_mean*100)

        # np.save(f'{home}/stuff/data/{year}/error_{year}_{month}.npy',
        #         extra_diff_mean)
        # np.save(f'{home}/stuff/data/{year}/avg_error_{year}_{month}.npy',
        #         avg_diff)
        np.save(f'{home}/stuff/data/{year}/cs_error_{year}_{month}.npy',
                extra_diff_mean)
        np.save(f'{home}/stuff/data/{year}/cs_avg_error_{year}_{month}.npy',
                avg_diff)

        
        i += 1
        
# %%
        
avg_diff_month = avg_diff
avg_diff_year = np.zeros((4, 8461))
for year in ['2011', '2013', '2016', '2017']:
    for month in ['01', '03', '05', '07', '09', '11']:
        avg_diff_year[i, :] = np.mean(avg_diff_month[6*(i):6*(i+1)], axis=0)
        plt.plot(chan, avg_diff_year[i])

        
        # %%
plt.figure()
extra_diff_mean_year = np.zeros((4, 8461))
for i in range(extra_diff_mean_year.shape[0]):
    extra_diff_mean_year[i, :] = np.mean(extra_diff_mean_month[6*(i):6*(i+1)], axis=0)
    plt.plot(chan, extra_diff_mean_year[i])

TAS = [297.71, 297.92, 298.30, 298.11]

np.save('/mnt/lustre01/pf/zmaw/u301023/iasi/stuff/data/error_all_years.npy',
        extra_diff_mean_year + 1)


#fb = scipy.stats.linregress(TAS, )

# %%
globtropextra = np.zeros((24, 8461))

plt.figure()
i = 0
for year in ['2011', '2013', '2016', '2017']:
    for month in ['01', '03', '05', '07', '09', '11']:
        globtropextra[i] = np.load(
            f'{home}/stuff/data/{year}/globtropextra_error_{year}_{month}.npy')
        plt.plot(chan[-8461:], globtropextra[i]*100)
        i += 1

plt.figure()
globtropextra_year = np.zeros((4, 8461))
for i in range(globtropextra_year.shape[0]):
    globtropextra_year[i, :] = np.mean(globtropextra[6*(i):6*(i+1)], axis=0)
    plt.plot(chan[-8461:], globtropextra_year[i]*100)

# np.save(f'{home}/stuff/data/globtropextra_error_all_years.npy',
#         globtropextra_year)

# %%
globtropextra = np.zeros((24, 8461))

plt.figure()
i = 0
for year in ['2011', '2013', '2016', '2017']:
    for month in ['01', '03', '05', '07', '09', '11']:
        globtropextra[i] = np.load(
            f'{home}/stuff/data/{year}/cs_error_{year}_{month}.npy')
        plt.plot(chan[-8461:], globtropextra[i]*100)
        i += 1

plt.figure()
globtropextra_year = np.zeros((4, 8461))
for i in range(globtropextra_year.shape[0]):
    globtropextra_year[i, :] = np.nanmean(globtropextra[6*(i):6*(i+1)], axis=0)
    plt.plot(chan[-8461:], globtropextra_year[i]*100)

np.save(f'{home}/stuff/data/cs_error_all_years.npy',
        globtropextra_year)


# %%
# clear sky: extra = glob - trop
a, b, c = 3, 12, 21


# set domains (name[a] = weighted_sum(name[b], name[c])
a, b, c = 0, 9, 18
            
global_area = np.zeros((1, n, 1))
global_area[0, :, 0] = area[0, :, 0]

diff = ((flux[b]*area[b] + flux[c]*area[c])/area[a] - flux[a])/flux[a]
#diff = ((flux[a]*area[a] - flux[b]*area[b])/area[c] - flux[c])/flux[a]


maxi = np.max(diff, axis=1)

chan = np.load('/work/um0878/user_data/froemer/rare_mistral/data/chan', allow_pickle=True)
ichan = chan[-8461:]

#plt.close('all')

avg = np.average(flux, weights=global_area.flatten(), axis=1)
avg_area = np.average(area, weights=global_area.flatten(), axis=1)
avg_diff = ((avg[b]*avg_area[b] + avg[c]*avg_area[c])/avg_area[a] - avg[a])/avg[a]
avg_max = np.max(avg_diff)

mean_diff = np.nanmean(diff, axis=0)
mean_diff2 = np.average(diff, weights=area[a].flatten(), axis=0)


#plt.figure()
#plt.xlim(ichan[0], ichan[-1])
#plt.hlines(0, ichan[0], ichan[-1], color='k')
#plt.plot(ichan, diff.T*100, color='grey', alpha=0.5, linewidth=0.1)
#plt.plot(ichan, mean_diff*100, color='red',
#         label=f'mean')
#plt.xlabel('wavenumber (cm-1)', fontsize=14)
#plt.ylabel('relative error (%)', fontsize=14)
#plt.title(name[c] + ' = ' '\n'  + name[a] + ' - ' + name[b])

plt.figure()
plt.xlim(ichan[0], ichan[-1])
plt.hlines(0, ichan[0], ichan[-1], color='k')
#plt.plot(ichan, diff.T*100, color='grey', alpha=0.5, linewidth=0.1)
plt.plot(ichan, mean_diff*100, color='red',
         label='orbit by orbit inference, then averaged')
plt.plot(ichan, avg_diff*100, color='blue',
         label='average first, then inference')
plt.xlabel('wavenumber (cm-1)', fontsize=14)
plt.ylabel('relative error (%)', fontsize=14)
plt.title(year + '/' + month + ': \n' + name[c] + ' = ' '\n'  + name[a] + ' - ' + name[b])

# the error is smaller if the fluxes for the not resolved domains are calculated
# for every domain separately and only then averaged
#plt.figure()
#
#plt.plot(ichan, 100* diff.mean(axis=0))
#plt.plot(ichan, 100*mean_diff)

# %%

#plt.figure()
#plt.xlim(ichan[0], ichan[-1])
#plt.hlines(0, ichan[0], ichan[-1], color='k')
#plt.plot(ichan, extra_diff.T*100, color='grey', alpha=0.5, linewidth=0.1)
#plt.plot(ichan, extra_diff_mean*100, color='red', label=f'mean')
#plt.title(f'sum - {name[index]}')
#plt.xlabel('wavenumber (cm-1)', fontsize=14)
#plt.ylabel('relative error (%)', fontsize=14)

plt.figure()
plt.xlim(ichan[0], ichan[-1])
plt.hlines(0, ichan[0], ichan[-1], color='k')
plt.plot(ichan, extra_diff_mean*100, color='red', label='mean')
plt.title(f'sum - {name[index]}')
plt.xlabel('wavenumber (cm-1)', fontsize=14)
plt.ylabel('relative error (%)', fontsize=14)

# %%
import scipy.stats

plt.figure()
t500 = np.array((266.08, 266.74, 267.09))
error = np.zeros((3, 8461))
derr = np.zeros((8461))
for y, year_month in enumerate(['2011_01', '2013_05', '2016_09']):
    error[y] = np.load(f'stuff/error_{year_month}.npy')
    plt.plot(ichan, 100*error[y], label=year_month)
plt.xlim(ichan[0], ichan[-1])
plt.hlines(0, ichan[0], ichan[-1], color='k')
plt.title(f'sum - {name[index]}')
plt.xlabel('wavenumber (cm-1)', fontsize=14)
plt.ylabel('relative error (%)', fontsize=14)
plt.legend()

for i in range(error.shape[1]):
    derr[i] = scipy.stats.linregress(t500, error.T[i])[0]

plt.figure()
plt.plot(ichan, derr*100)


# %%
name = np.array(name)
name2 = name.reshape((3, 3, 3))

flux2 = flux.reshape((3, 3, 3, 430, 8461))
area2 = area.reshape((3, 3, 3, 430, 1))

