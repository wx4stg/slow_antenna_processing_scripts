import numpy as np
import xarray as xr
import datetime
from glob import glob
from scipy.signal import medfilt
import pandas as pd
from datetime import datetime
import os
import sa_common
import argparse


SAMPLE_RATE = 9600  # Hertz
u4max = 4294967295


def compress_all(nc_grids, min_dims=1):
    for var in nc_grids:
        if len(nc_grids[var].dims) >= min_dims:
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
    ds['time'] = ds.time_uncorrected - (correction*1e9).astype('timedelta64[ns]')
    return ds
    
    
def add_other_time_vars(ds):
    ds['dt_system'] = (ds.time_uncorrected-ds.time_uncorrected[0]).astype('datetime64[ns]').astype('f8')/1e9
    ds['dt_adc'] = (ds.pps_micro - ds.pps_micro[0]).astype('f8')/1e6
    return ds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Slow Antenna .raw data')
    parser.add_argument('-i', '--input', nargs='+', required=True, help='Path or paths to slow antenna files to convert.')
    parser.add_argument('-s', '--sensor-num', required=True, help='The ADC number of the sensor that collected the data.', type=int)
    parser.add_argument('--latitude', type=float, help='Latitude of the sensor. Overridden if the latitude is present in the file name.')
    parser.add_argument('--longitude', type=float, help='Longitude of the sensor. Overridden if the longitude is present in the file name.')
    parser.add_argument('--altitude', type=float, help='Altitude of the sensor. Overridden if the altitude is present in the file name.')
    parser.add_argument('-r', '--relay', help='Relay used (a/b/c). Overridden if the relay is present in the file name.', type=str)
    parser.add_argument('-o', '--output', required=True, help='Directory to save netCDF output files.')
    args = parser.parse_args()

    files = args.input
    if len(files) == 1:
        files = glob(files[0])
    else:
        files = files
    files = sorted(files)

    # Count of all rollovers
    total_rollovers = 0
    for i, filepath in enumerate(files):
        sensor_lat = args.latitude if args.latitude else np.nan
        sensor_lon = args.longitude if args.longitude else np.nan
        sensor_alt = args.altitude if args.altitude else np.nan
        sensor_relay = args.relay if args.relay else np.nan
        filename = os.path.basename(filepath)
        fn_split = filename.replace('.raw', '').split('_')
        if len(fn_split) == 2:
            # This is an "old" rawfile
            T0 = datetime.strptime(fn_split[0]+"_"+fn_split[1], "%Y%m%d%H%M%S_%f")
        elif len(fn_split) == 3:
            # This is a "March 2024" rawfile
            T0 = datetime.strptime(fn_split[0]+"_"+fn_split[1], "%Y%m%d%H%M%S_%f")
            sensor_relay = fn_split[2]
        elif len(fn_split) == 9:
            # This is a "July 2024" rawfile
            T0 = datetime.strptime(fn_split[0]+"_"+fn_split[1]+"_"+fn_split[2], "%Y%m%d_%H%M%S_%f")
            sensor_lat = float(fn_split[3]) if fn_split[3] != 'NO' else np.nan
            sensor_lon = float(fn_split[4]) if fn_split[4] != 'FIX' else np.nan
            sensor_alt = float(fn_split[5]) if fn_split[5] != '2Donly' else np.nan
            if sensor_alt == 0.0:
                sensor_alt = np.nan
            sensor_relay = fn_split[8]
        else:
            raise ValueError(f'Unknown format of file: {filename}')

        # Read and decode raw data
        if i == 0:
            raw_data = sa_common.read_SA_file(filepath)
        else:
            raw_data = sa_common.read_SA_file(filepath, previous_file=files[i-1])
        adc_pps_micros, adc = sa_common.decode_SA_array(raw_data)
        adc_ready, new_rolls = correct_micros(adc_pps_micros, SAMPLE_RATE)
        # Add on the cumulative rollovers from previous files
        adc_ready += total_rollovers*u4max
        total_rollovers += new_rolls

        time_orig = np.array([T0]).astype('datetime64[ns]')[0] + (adc_ready-adc_ready[0]).astype('timedelta64[us]')
            
        ds = xr.Dataset(pd.DataFrame(
            {'ADC':adc,
            'pps_micro':adc_ready,
            'time_uncorrected':time_orig})).reset_index('dim_0').drop_vars('dim_0').rename_dims({'dim_0':'sample'})
        
        ds = interpolate_across_system_times(add_other_time_vars(ds))
        fileoutname = f'SA{args.sensor_num}_{T0.strftime("%Y-%m-%d_%H-%M-%S")}.nc'
        ds = ds.assign_coords({'sensor_num' : [args.sensor_num]})
        match sensor_relay:
            case 'a':
                sensor_relay = 0
            case 'b':
                sensor_relay = 1
            case 'c':
                sensor_relay = 2
            case _:
                raise ValueError(f'Unknown relay: {sensor_relay}')
        ds = ds.assign(
            lat = ('sensor_num', np.array([sensor_lat])),
            lon = ('sensor_num', np.array([sensor_lon])),
            alt = ('sensor_num', np.array([sensor_alt])),
            relay = ('sensor_num', np.array([sensor_relay]))
        )
        ds['ADC'].attrs['long_name'] = 'Slow antenna ADC reading'
        ds['pps_micro'].attrs['long_name'] = 'ADC local reference clock time, corrected for rollover'
        ds['pps_micro'].attrs['units'] = 'microseconds'
        ds['time_uncorrected'].attrs['long_name'] = 'Time reported by the ADC clock'
        ds['time'].attrs['long_name'] = 'Time of the ADC sample in UTC, corrected for system time errors'
        ds['dt_system'].attrs['long_name'] = 'Time offset of the Linux system clock, relative to the first sample'
        ds['dt_system'].attrs['units'] = 'seconds'
        ds['dt_adc'].attrs['long_name'] = 'Time offset of the ADC local reference clock, relative to the first sample'
        ds['dt_adc'].attrs['units'] = 'seconds'
        ds['lat'].attrs['long_name'] = 'Latitude of the sensor'
        ds['lon'].attrs['long_name'] = 'Longitude of the sensor'
        ds['alt'].attrs['long_name'] = 'Altitude of the sensor'
        ds['relay'].attrs['long_name'] = 'Active relay of the ADC, 0=a, 1=b, 2=c'
        comp_ds = compress_all(ds)
        comp_ds.to_netcdf(os.path.join(args.output, fileoutname))
