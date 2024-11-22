#!/usr/bin/env python
# coding: utf-8

# In[49]:


from time import sleep
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import struct
import datetime
import sys
from glob import glob
import scipy.signal as signal
from scipy.signal import medfilt
import pandas as pd
import datetime
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
from matplotlib.ticker import FormatStrFormatter


# In[3]:


get_ipython().run_line_magic('matplotlib', 'widget')


# In[50]:


def interpolate_across_system_times(ds, SAMPLE_RATE=10000):
    
    nominal_dt = 1.0/SAMPLE_RATE
    good_sample_thresh = (3*nominal_dt)

    bad_adc = (ds['ADC'].values < 0.0) 
    adc_minus_system = (ds['dt_adc']-ds['dt_system']).values
    adc_minus_system[bad_adc] = np.nan
    jump = np.diff(adc_minus_system, prepend=np.nan)

    # Also blank out the ADC and ADC microseconds where we have bad data.
    ds['dt_adc'] = ds.dt_adc.where(~bad_adc, other=np.nan)
    ds['dt_system'] = ds.dt_system.where(~bad_adc, other=np.nan)
    
    is_jump = np.abs(jump) > good_sample_thresh
    jump_idx, = np.where(is_jump)
    # Include first and last points
    jump_idx = np.insert(jump_idx, [0,jump_idx.shape[0]], [0,adc_minus_system.shape[0]-1])
    jump = jump[jump_idx]
    
    all_samples = np.arange(adc_minus_system.shape[0])
    offset_curve = np.interp(all_samples, 
                             jump_idx,
                             adc_minus_system[jump_idx])
    correction = offset_curve - adc_minus_system
    ds['time'] = ds.time_orig_method - (correction*1e9).astype('timedelta64[ns]')
    return ds
def plot_one_SA(ds):
    t_axis_data = ds['dt_adc']

    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].plot(t_axis_data,(ds['dt_adc']-ds['dt_system']))
    ax[0].set_xlabel('ADC clock elapsed (s)')
    ax[0].set_ylabel('System clock - ADC clock (s)')
    bits_to_volts = (5/((2**24)-1))
    ax[1].plot(t_axis_data, medfilt(ds['ADC'].values, 7)*bits_to_volts)
    ax[1].set_xlabel('ADC clock elapsed (s)')
    ax[1].set_ylabel('ADC (DN)')

    fig.tight_layout()


def plot_corrected(dss, labels):
    n_datasets = len(dss)
    fig, ax = plt.subplots(n_datasets,1, sharex=True, sharey=False)
    bits_to_volts = (5/((2**24)-1))
    for axi, (ds, label) in enumerate(zip(dss, labels)):
        t = ds['time']# + np.timedelta64('5', 'h')
        adc = medfilt(ds['ADC'].values, 7)*bits_to_volts
        # adc = ds['ADC'].values*bits_to_volts
        ax[axi].plot(t, adc)
        # ax[axi].set_ylabel('ADC (Volts)',fontsize = 10)
        ax[axi].set_title(label)
        # ax[axi].set_ylim(4e6,6e6)
        # ax[axi].set_ylim(1.0,2.0)
        # ax[axi].set_ylim(np.nanmin(adc) - 0.01, np.nanmax(adc) + 0.01 )
        ax[axi].set_ylim(np.nanmean(adc) - 0.1, np.nanmax(adc) + 0.01 )
        ax[axi].set_xlim(t[0],t[-2])
    ax[-1].set_xlabel('Time (UTC)')
    # ax[-1].set_ylabel('ADC (Volts)',fontsize = 10)
    # plt.ylabel("common Y")
    fig.supylabel('ADC (Volts)')
    plt.xticks(rotation=45)
    locator = mdates.AutoDateLocator(minticks = 3, maxticks = 7)
    formatter = mdates.ConciseDateFormatter(locator)

    ax[-1].xaxis.set_major_locator(locator)
    ax[-1].xaxis.set_major_formatter(formatter)
    
    # plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    return fig

plt.close('all')


# In[53]:


# ds1 = xr.open_mfdataset(sorted(glob('./202306*-sensor1-*.nc')), combine='nested', concat_dim='sample')
# ds3 = xr.open_mfdataset(sorted(glob('./202306*-sensor3-*.nc')), combine='nested', concat_dim='sample')
# ds8 = xr.open_mfdataset(sorted(glob('./202306*-sensor8-*.nc')), combine='nested', concat_dim='sample')
date = '20240829'
date2 = '20240830'
hour = '20'
minute='0*'
# ds1 = xr.open_mfdataset(sorted(glob('/Volumes/Extreme_SSD/SA_SYNC/{0}/processed/sensor_01/{1}{2}{3}.nc'.format(date,date2,hour,minute))), combine='nested', concat_dim='sample')
ds2 = xr.open_mfdataset(sorted(glob('/Volumes/Extreme_SSD/SA_SYNC/{0}/processed/sensor_02/{1}{2}{3}.nc'.format(date,date2,hour,minute))), combine='nested', concat_dim='sample')
# ds3 = xr.open_mfdataset(sorted(glob('/Volumes/Extreme_SSD/SA_SYNC/{0}/processed/sensor_03/{1}{2}{3}.nc'.format(date,date2,hour,minute))), combine='nested', concat_dim='sample')
# ds4 = xr.open_mfdataset(sorted(glob('/Volumes/Extreme_SSD/SA_SYNC/{0}/processed/sensor_04/{1}{2}{3}.nc'.format(date,date2,hour,minute))), combine='nested', concat_dim='sample')
# ds5 = xr.open_mfdataset(sorted(glob('/Volumes/Extreme_SSD/SA_SYNC/{0}/processed/sensor_05/{1}{2}{3}.nc'.format(date,date2,hour,minute))), combine='nested', concat_dim='sample')
# ds6 = xr.open_mfdataset(sorted(glob('/Volumes/Extreme_SSD/SA_SYNC/{0}/processed/sensor_06/{1}{2}{3}.nc'.format(date,date2,hour,minute))), combine='nested', concat_dim='sample')
# ds7 = xr.open_mfdataset(sorted(glob('/Volumes/Extreme_SSD/SA_SYNC/{0}/processed/sensor_07/{1}{2}{3}.nc'.format(date,date2,hour,minute))), combine='nested', concat_dim='sample')
ds8 = xr.open_mfdataset(sorted(glob('/Volumes/Extreme_SSD/SA_SYNC/{0}/processed/sensor_08/{1}{2}{3}.nc'.format(date,date2,hour,minute))), combine='nested', concat_dim='sample')
ds9 = xr.open_mfdataset(sorted(glob('/Volumes/Extreme_SSD/SA_SYNC/{0}/processed/sensor_09/{1}{2}{3}.nc'.format(date,date2,hour,minute))), combine='nested', concat_dim='sample')
# ds10 = xr.open_mfdataset(sorted(glob('/Volumes/Extreme_SSD/SA_SYNC/{0}/processed/sensor_10/{1}{2}{3}.nc'.format(date,date2,hour,minute))), combine='nested', concat_dim='sample')

# ds11 = xr.open_mfdataset(sorted(glob('/Volumes/Extreme_SSD/SA_SYNC/{0}/processed/sensor_11/{1}{2}{3}.nc'.format(date,date2,hour,minute))), combine='nested', concat_dim='sample')



def add_other_time_vars(ds):
    ds['dt_system'] = (ds.time_orig_method-ds.time_orig_method[0]).astype('datetime64[ns]').astype('f8')/1e9
    ds['dt_adc'] = (ds.pps_micro - ds.pps_micro[0]).astype('f8')/1e6
    return ds

# ds1 = interpolate_across_system_times(add_other_time_vars(ds1))
# ds2 = interpolate_across_system_times(add_other_time_vars(ds2))
# ds3 = interpolate_across_system_times(add_other_time_vars(ds3))
# ds4 = interpolate_across_system_times(add_other_time_vars(ds4))
# ds5 = interpolate_across_system_times(add_other_time_vars(ds5))
# ds6 = interpolate_across_system_times(add_other_time_vars(ds6))
# ds7 = interpolate_across_system_times(add_other_time_vars(ds7))
# ds8 = interpolate_across_system_times(add_other_time_vars(ds8))
# ds9 = interpolate_across_system_times(add_other_time_vars(ds9))
# ds11 = interpolate_across_system_times(add_other_time_vars(ds11))
plt.close('all')
# fig = plot_corrected([ds2,ds3,ds5,ds6, ds7,ds8,ds10], ['SA2','SA3','SA5','SA6', 'SA7','SA8','SA10'])
fig = plot_corrected([ds2,ds8,ds9], ['SA2','SA8','SA9'])
# fig = plot_corrected([ds6,ds7,ds8], ['SA6','SA7','SA8'])
# fig = plot_corrected([ds1,ds2,ds5], ['SA1','SA2','SA5'])


# In[44]:


ds2['time'].values[-2]


# In[67]:


# plt.close('all')
# # fig = plot_corrected([ds2,ds3,ds5,ds6, ds7,ds8,ds10], ['SA2','SA3','SA5','SA6', 'SA7','SA8','SA10'])
# fig = plot_corrected([ds1,ds2,ds5], ['SA1','SA2','SA5'])


# In[5]:


plot_one_SA(ds7)


# In[14]:


bits_to_volts = (5/((2**24)-1))
np.nanmean(medfilt(ds7['ADC'].values, 7)*bits_to_volts)


# In[55]:


# plot_one_SA(ds2)


# In[53]:


# plot_one_SA(ds3)


# In[54]:


# plot_one_SA(ds5)


# In[56]:


# plot_one_SA(ds6)


# In[57]:


# plt.close('all')
# plot_one_SA(ds3)


# In[18]:


fig = plot_corrected([ds1, ds8,ds10], ['SA1', 'SA8','SA10'])


# In[24]:


fig = plot_corrected([ds3,ds5], ['SA3','SA5'])


# # Development of the correction method to interpolate across system times
# 
# As implemented above.

# In[21]:


SAMPLE_RATE = 10000
nominal_dt = 1.0/SAMPLE_RATE

good_sample_thresh = (3*nominal_dt)



adc_micro_jump = np.diff(ds8['dt_adc'].values, prepend=0)
system_jump = np.diff(ds8['dt_system'].values, prepend=0)


huge_adc_micro_jump = np.abs(adc_micro_jump) > 100.0
bad_adc = (ds8['ADC'].values < 0.0) 
errs = (# huge_adc_micro_jump &
         bad_adc
       )


any_adc_micro_jump = np.abs(adc_micro_jump) > good_sample_thresh
any_system_jump = np.abs(system_jump) > good_sample_thresh


adc_minus_system = (ds8['dt_adc']-ds8['dt_system']).values
adc_minus_system[errs] = np.nan

adc_minus_system_jump = np.diff(adc_minus_system, prepend=np.nan)


# In[23]:


jump = adc_minus_system_jump.copy()
is_jump = np.abs(adc_minus_system_jump) > good_sample_thresh
jump_idx, = np.where(is_jump)
# Include first and last points
jump_idx = np.insert(jump_idx, [0,jump_idx.shape[0]], [0,adc_minus_system.shape[0]-1])
jump = jump[jump_idx]
print(jump_idx)
print(jump)
print(adc_minus_system[jump_idx])


# In[26]:


all_samples = np.arange(adc_minus_system.shape[0])
offset_curve = np.interp(all_samples, 
                         jump_idx,
                         adc_minus_system[jump_idx])


# In[14]:


plt.close('all')
fig, ax = plt.subplots(2,1, sharex=True)
ax[0].plot(adc_minus_system)
ax[0].plot(offset_curve)
ax[0].set_ylabel('Accumulated ADC clock excess (s)')
ax[0].set_xlabel('Sample')

ax[1].plot(jump_idx, jump,'.')
ax[1].set_ylabel('Step increment (s)')
ax[1].set_xlabel('Sample')


# We assume the system clock $s$ is basically correct, while the ADC clock $a$ drifts.
# 
# At the start of each file $i$ the system clock is at $s_i$ and the ADC clock at $a_i$. The previous script calculates the `time_orig_method` variable as $s_a = s_i + (a - a_i)$. 
# 
# The top curve above is $a - s_a = a - s_i - (a - a_i) = a_i - s_i$, which at the start of each file accumulates another increment of error $e_i = a_i - s_i$. If the ADC clock is fast, $e_i$ will be positive, so we need to subtract this amount off the system clock.
# 
# The interpolation curve above gives us a correction to the free-running ADC clock, so that we can correct $s_a$ (`time_orig_method`). 

# In[30]:


correction = offset_curve - adc_minus_system
time = ds8.time_orig_method - (correction*1e9).astype('timedelta64[ns]')


# In[32]:


dt_corrected = time - ds8.time_orig_method


# In[37]:


# The system times shouldn't have changed.
assert np.allclose(dt_corrected[jump_idx].values.astype(int), 0)


# In[ ]:




