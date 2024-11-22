#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import csv 
from pandas.core.common import flatten


# In[2]:


pip install dask


# In[3]:


SAMPLE_RATE = 10000  # Hertz
u4max = 4294967295

# SAPATH = "/Users/coogray/Documents/SA_DATA/Jun_15_2023/"
# SAPATH = "//Users/kelcy/DATA/20240710/"
SAPATH="/Volumes/Extreme_SSD/SA_SYNC/20240902/"
SAOUT = "/Volumes/Extreme_SSD/SA_SYNC/20240829/"


# In[4]:


import time

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


# In[5]:


def convert_adc_to_decimal(value):
    modulo = 1 << 24
    max_value = (1 << 23) - 1
    if value > max_value:
        value -= modulo
    return value
SERIAL_SPEED = 2000000


# In[6]:


def decode_data_packet(mp):
    # print(mp)
    result = dict()
    result['start_byte'] = struct.unpack('B', mp[0:1])[0]
    result['b1'] = struct.unpack('B', mp[1:2])[0]
    result['b2'] = struct.unpack('B', mp[2:3])[0]
    result['b3'] = struct.unpack('B', mp[3:4])[0]
    result['adc_pps_micros'] = struct.unpack('I', mp[4:8])[0]
    result['end_byte'] = struct.unpack('B', mp[8:9])[0]
    adc_hex = mp[1:4].hex()
    adc_ba = bytearray()
    adc_ba += mp[1:2]
    adc_ba += mp[2:3]
    adc_ba += mp[3:4]
    adc_ba += b'\x00'

    # print(mp[3:4])
    # print(adc_ba)
    adc_reading = struct.unpack('>i', adc_ba[:])[0]

    adc_reading = mp[1]
    adc_reading = (adc_reading << 8) | mp[2]
    adc_reading = (adc_reading << 8) | mp[3]
    adc_reading = convert_adc_to_decimal(adc_reading)


    result['adc_reading'] = adc_reading
    return result
def notch_sixty(s, fs):
    f0 = 60.0  # Frequency to be removed from signal (Hz)
    Q = 2.0  # Quality factor = center / 3dB bandwidth
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, s)

# def notch_sixty(s, fs):
#     f0 = 0.125  # Frequency to be removed from signal (Hz)
#     Q = 0.5 #2.0  # Quality factor = center / 3dB bandwidth
#     b, a = signal.iirnotch(f0, Q, fs)
#     return signal.filtfilt(b, a, s)


# # Preprocessing - raw to NetCDF

# In[ ]:





# In[7]:


def compress_all(nc_grids, min_dims=1):
    for var in nc_grids:
        if len(nc_grids[var].dims) >= min_dims:
            # print("Compressing ", var)
            nc_grids[var].encoding["zlib"] = True
            nc_grids[var].encoding["complevel"] = 5
            nc_grids[var].encoding["contiguous"] = False
    return nc_grids


def correct_micros(micros, fs, tol=0.1, window=7):
    """
    micros is the timing data (integer microseconds) to be corrected.
    fs is the nominal sample rate in samples/second
    tol is the fraction of the sample interval used in defining large errors
    window is the number of samples used to correct single-point noise.
    
    Output is corrected so that the time change between samples is
    not greater than tol/fs.
    
    Two things can result in time that does not increase steadily.
    1. Random read errors, which may be positive or negative. 
       These are embedded in a steadily increasing ramp, so the 
       median of these samples (for a large enough window)
       Within window we expect these erros to be significantly differnt from the median
    2. Integer rollover, which results in an instananeous change of -u4_int_max
        
    First, run a median filter to eliminate most spikes. If the number of errors in the
    window is larger than (window-1)/2, then the window will have an incorrect value.
    But this value will be much smaller than the integer max, allowing us to isolate the
    truly large negative changes from integer rollovers.
    
    >>> test_ramp = np.arange(-25, 25).astype('u4') + u8max
    >>> medfilt(test_ramp, 3)
    Near the jump this code gives 
    [4294967292, 4294967293, 4294967294, 4294967294, 1, 1, 2, 3]
    i.e., the delta remains large.
        
    At the edges of the dataset, we have to be more careful. 
    After correcting the middle of the dataset, check for bad data one more time.
    If found, simply extrapolate using the floor of the mean sample rate.
    
    """
    micros_orig = micros
    
    nominal_dt_micro = int(1e6/fs)
    
    def get_bad(m):
        change = np.diff(m)
        return (change < 0) & (change > int(tol*nominal_dt_micro))

     
    # Median filter, and promote to 64 bit int
    med_micro = medfilt(micros, window).astype('i8')
    
    ### Detect the large negatives, and add those to the original data
    
    # After median filtering, the jump could be corrupted by noise.
    # Allow some tolerance
    big_neg_thresh = -int((1-tol)*u4max)
    sample_delta = np.diff(med_micro)
    # print(big_neg_thresh, "negative rollover threshold; ", sample_delta.min(), "lowest delta")
    large_neg = np.hstack([[False], 
                           (sample_delta < big_neg_thresh)])
    n_rollovers = large_neg.sum()
    # print(n_rollovers, "total rollovers")
    
    # Set the points with large negative deltas to uint4max,
    # then cumsum will accumulate correction amount to all points
    # after each detected jump.
    correction_mark = np.zeros_like(micros)
    correction_mark[large_neg] = u4max
    neg_correct = np.cumsum(correction_mark)
    
    micros = med_micro + neg_correct
    
    ### Check for remaining problems

    bad = get_bad(micros)

    if bad.sum() > 0:
        # medfilt one more time
        micros = medfilt(micros, window)
    else:
        return micros, n_rollovers

    bad = get_bad(micros)
    
    # Check for problems at the beginning and end
    front = slice(0, window)
    if bad[front].sum() > 0:
        micros[front] = micros[window] + nominal_dt_micro*np.arange(-window, 0, 1)

    back = slice(-window, None)
    if bad[back].sum() > 0:
        micros[back] = micros[-window] + nominal_dt_micro*np.arange(0, window, 1)
    
    # Now look at any remaning deltas
    bad = get_bad(micros)

    if bad.sum() > 0:
        # medfilt one more time with a very large window
        micros = medfilt(micros, window*3)
        bad = get_bad(micros)
        if bad.sum() > 0:
            raise ValueError("Could not correct all time errors; total bad = {0}".format(bad.sum()))
        else:
            return micros, n_rollovers
    else:
        return micros, n_rollovers

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
    
    
def add_other_time_vars(ds):
    ds['dt_system'] = (ds.time_orig_method-ds.time_orig_method[0]).astype('datetime64[ns]').astype('f8')/1e9
    ds['dt_adc'] = (ds.pps_micro - ds.pps_micro[0]).astype('f8')/1e6
    return ds


# In[63]:


# T0 = pd.to_datetime(datetime.strptime(os.path.basename(filename)[:-43],"%Y%m%d_%H%M%S_%f"))
# print(T0)
os.path.basename(filename)[:22]#[:-43]


# In[13]:


#modified 7/27/24 to account for _a/b/c in the file name.
#VERSION FOR INCLUDING THE UNCORRECTED TIME VIA THE FILE NAME
#time correction script
#purpose: add n x 4294967295 each time the pps maxes out

# import os
# out  = [os.rename(x, x[:-4]+'_35.248810N_97.594849W_100A.raw') for x in files1]

#step 1 read in the first file, save that as T0
#find delta t between the first sample of the next file and the last sample of the first file, if it is inasnely large, +1
#the count and repeat

path1 = os.path.join(SAPATH, "sensor_07")
files1 = sorted(glob(path1+'/'+"20240*.raw")), '07'
path2 = os.path.join(SAPATH, "sensor_08")
files2 = sorted(glob(path2+'/'+"20240*.raw")), '08'
path3 = os.path.join(SAPATH, "sensor_09")
files3 = sorted(glob(path3+'/'+"20240*.raw")), '09'

files, sensorname = files2
# SAOUTPATH = os.path.join(SAOUT,"processed",("sensor_"+sensorname))

# Resets for each hour chunk of files
all_data = []
# Count of all rollovers
total_rollovers = 0
# First adc_ready value in the whole sequence 
prev_hr = None
do_write = False
for _idx, filename in enumerate(sorted(files)):
    print(filename)
    # T0 = pd.to_datetime(datetime.strptime(os.path.basename(filename)[:-6], "%Y%m%d%H%M%S_%f"))
    T0 = pd.to_datetime(datetime.strptime(os.path.basename(filename)[:22],"%Y%m%d_%H%M%S_%f"))
    print(T0)
    if prev_hr is None: prev_hr = T0.hour

    # Read and decode raw data
    data_raw_packets=[]
    data_start_bytes = []
    data_packet_length = 8
    data_packets = []
    this_packet_length = data_packet_length + 1
    with open(filename, mode = 'rb') as file:
        ba = file.read()
    for i in range(len(ba) - data_packet_length):
        if (ba[i] == 190) and (ba[i+data_packet_length] == 239):
            data_start_bytes.append(i)
    data_raw_packets.extend([ba[sb:sb+this_packet_length] for sb in data_start_bytes[:-1]])
    data_packets = [decode_data_packet(b) for b in data_raw_packets]

    # Detect negative steps in this file, and cleanup noise spikes in ADC's time counter
    adc_ready, new_rolls = correct_micros(np.asarray([dp['adc_pps_micros'] for dp in data_packets]),
                               SAMPLE_RATE)
    # Add on the cumulative rollovers from previous files
    adc_ready += total_rollovers*u4max
    total_rollovers += new_rolls

    print(adc_ready[0])
    print(adc_ready[-1])
    time_orig = T0 + (adc_ready-adc_ready[0]).astype('timedelta64[us]')

    # Sensor measurements from the ADC. 24 bit sensor, so 32 bit int will be fine.
    adc = np.asarray([dp['adc_reading'] for dp in data_packets]).astype('int32')
        
    ds = xr.Dataset(pd.DataFrame(
        {'ADC':adc,
         'pps_micro':adc_ready,
         'time_orig_method':time_orig})).reset_index('dim_0').drop_vars('dim_0').rename_dims({'dim_0':'sample'})
    
    ds = interpolate_across_system_times(add_other_time_vars(ds))
    
    # all_data.append(ds)
    
    # if T0.hour != prev_hr:
    #     prev_hr = T0.hour
    #     print('hour roll over')
    #     do_write = True

    # last_hour_chunk = datetime(T0.year, T0.month, T0.day, T0.hour) - timedelta(hours=1)
    # # data will eventually go in this file ...
    # last_hour_string = last_hour_chunk.strftime('%Y%m%d_%H')
    # fileoutname = last_hour_string+"_"+sensorname+"_"+os.path.basename(files[0])[:-4]+".nc"
    
    # fileoutname = T0.strftime("%Y%m%d%H%M%S_%f")+"-"+sensorname+"-"+os.path.basename(filename)[0][:-4]+".nc"
    fileoutname = T0.strftime("%Y%m%d%H%M%S_%f")+"_"+sensorname+"_"+os.path.basename(filename)[23:-4]+".nc"
    comp_ds = compress_all(ds)
    # print(comp_ds)
    # comp_ds.to_netcdf(os.path.join(SAOUTPATH,'/', fileoutname,))
    comp_ds.to_netcdf(fileoutname)


    # if do_write:
    #     # Write all but the last chunk read, and carry that over to the next hour segment.
    #     ds_all = xr.concat([d.reset_index('dim_0').drop('dim_0') for d in all_data[:-1]], dim='dim_0').rename_dims({'dim_0':'sample'})
    #     compress_all(ds_all).to_netcdf(fileoutname)
    #     all_data = all_data[-1:]
    #     print(all_data[0].time_orig_method.values[0])
    #     do_write = False

# write any remaining data
# ds_all = xr.concat([d.reset_index('dim_0').drop('dim_0') for d in all_data], dim='dim_0').rename_dims({'dim_0':'sample'})
# compress_all(ds_all).to_netcdf(fileoutname)


# In[80]:


SAPATH


# In[25]:


os.path.basename(filename)[23:-4]+".nc"


# In[ ]:


ds = xr.open_mfdataset(sorted(glob('./202306*-sensor3-*.nc')), combine='nested', concat_dim='sample')


# In[19]:


ds


# In[ ]:


# %matplotlib widget
# plt.close('all')
# fig, axs = plt.subplots(2,1)
# ds.pps_micro.plot(ax=axs[0])
# ds.time_orig_method.plot(ax=axs[1])
# fig.tight_layout()


# In[ ]:


# ds['dt_system'] = (ds.time_orig_method-ds.time_orig_method[0]).astype('datetime64[ns]').astype('f8')/1e9
# ds['dt_adc'] = (ds.pps_micro - ds.pps_micro[0]).astype('f8')/1e6

# ds['system_sample_dt'] = (ds.time_orig_method-ds.time_orig_method[0]).astype('datetime64[ns]').astype('f8')/1e9
# ds['adc_sample_dt'] = (ds.pps_micro - ds.pps_micro[0]).astype('f8')/1e6

# # ds['dt_adc']=(dt_micro.values).astype('timedelta64[us]')


# In[ ]:


# %matplotlib widget
# fig, axs = plt.subplots(2,1)
# ds.dt_system.plot(ax=axs[0])
# ds.dt_adc.plot(ax=axs[1])
# # axs[2].plot(np.diff(ds.time_orig_method.astype('i8').values)/1e9)
# fig.tight_layout()


# In[ ]:


# t_axis_data = ds['dt_adc']

# fig, ax = plt.subplots(2,1, sharex=True)
# ax[0].plot(t_axis_data,(ds['dt_adc']-ds['dt_system']))
# ax[0].set_xlabel('ADC clock elapsed')
# ax[0].set_ylabel('System clock - ADC clock (s)')

# ax[1].plot(t_axis_data, medfilt(ds['ADC'].values, 7))
# ax[1].set_xlabel('ADC clock elapsed (s)')
# ax[1].set_ylabel('ADC (DN)')

# fig.tight_layout()


# In[ ]:


# ds


# In[ ]:




