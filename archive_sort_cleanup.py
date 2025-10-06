#!/usr/bin/env python3
# Sorts TTU Slow Antenna network data and prunes unininteresting data by LMA
# Created 14 September 2025 by Sam Gardner <samuel.gardner@ttu.edu> 

import argparse
from os import path
from glob import glob
from datetime import datetime as dt, timedelta
import numpy as np
import pandas as pd
import h5py
from pyxlma import coords

parser = argparse.ArgumentParser(description='Slow antenna archive pruning and sorting.')
parser.add_argument('--archive-root', '-r', type=str, required=True, default='/mnt/reservoir/SA_DATA/', help='Top level directory of slow antenna data archive.')
parser.add_argument('--lma-data', '-l', type=str, default=None, help='Directory containing LMA data for correlation with slow antenna data. On rime as of Oct. 2025, this is "/archive/lmaclimo/h5_files/". Leave unspecified to skip pruning.')
parser.add_argument('--dry-run', '-n', action='store_true', help='Log actions without making any changes.')
parser.add_argument('--cpu-serial-log', '-c', type=str, default=None, help='Path to a CSV file logging CPU serial numbers and their associated sensor numbers and deployment dates. Leave unspecified to skip.')

def delete_file(file_path, reason=None, dry_run=True):
    if reason is not None:
        reason_str = f' ({reason})'
    else:
        reason_str = ''
    if dry_run:
        print(f'Would remove {file_path}{reason_str}')
    else:
        print(f'Removing {file_path}{reason_str}')
        if path.isdir(file_path):
            rmdir(file_path)
        else:
            remove(file_path)

def move_file(old_path, new_path, reason=None, dry_run=True):
    if reason is not None:
        reason_str = f' ({reason})'
    else:
        reason_str = ''
    if dry_run:
        print(f'Would move {old_path} to {new_path}{reason_str}')
    else:
        print(f'Moving {old_path} to {new_path}{reason_str}')
        rename(old_path, new_path)

def unhandleable_file(message, dry_run=True):
    if dry_run:
        print(f'Would raise error: {message}')
    else:
        raise ValueError(message)

if __name__ == '__main__':
    args = parser.parse_args()

    archive_root = args.archive_root
    dry_run = args.dry_run

    if not dry_run:
        print('----> FILESYSTEM CHANGES CAN BE PERFORMED <----')
        from os import remove, rmdir, rename
        from time import sleep
        print('----> Press CTRL + C in the next 5 seconds to abort! <----')
        sleep(5)
    geosys = coords.GeographicSystem()
    date_directories = sorted(glob(path.join(archive_root, "*")))
    date_dirs_to_proc = []
    for date_dir in date_directories:
        try:
            this_date = dt.strptime(path.basename(date_dir), '%Y%m%d')
            date_dirs_to_proc.append(date_dir)
        except ValueError as e:
            if 'does not match format' in str(e):
                delete_file(date_dir, reason='first level directory does not conform to date format string YYYYMMDD/', dry_run=dry_run)
                continue
            else:
                raise e
    if args.cpu_serial_log is not None:
        cpu_df = pd.DataFrame(columns=[path.basename(d) for d in date_dirs_to_proc])
    for date_dir in date_dirs_to_proc:
        this_date = dt.strptime(path.basename(date_dir), '%Y%m%d')
        sensor_dirs = sorted(glob(path.join(date_dir, '*')))
        this_date_ids = dict()
        dts_this_day = []
        lons_this_day = []
        lats_this_day = []
        paths_this_day = []
        for sensor_dir in sensor_dirs:
            if path.basename(sensor_dir).startswith('sensor_') and len(path.basename(sensor_dir)) == 9:
                try:
                    sensor_num = int(sensor_dir[-2:])
                except Exception as e:
                    delete_file(sensor_dir, reason='second level directory does not conform to format sensor_XX', dry_run=dry_run)
            else:
                delete_file(sensor_dir, reason='second level directory does not conform to format sensor_XX', dry_run=dry_run)
            if str(sensor_num).zfill(2) != sensor_dir[-2:]:
                unhandleable_file(f'Directory {sensor_dir} does not conform to format sensor_XX, but I\'m not sure what to do with it', dry_run=dry_run)
            sensor_dir_content = sorted(glob(path.join(sensor_dir, '*')))
            sensor_dir_content_names = [path.basename(s) for s in sensor_dir_content]
            # -1 = invalid, 0 = SA_log.out, 1 = time only (old old), 2 = time, relay (old), 3 = time, location, CPU, GPS err, relay (current)
            filename_spec = np.zeros(len(sensor_dir_content), dtype=int)
            cpu_ids = np.zeros(len(sensor_dir_content), dtype=int)
            all_lons = np.full(len(sensor_dir_content), np.nan, dtype=float)
            all_lats = np.full(len(sensor_dir_content), np.nan, dtype=float)
            all_alts = np.full(len(sensor_dir_content), np.nan, dtype=float)
            for i, psbl_rawfile in enumerate(sensor_dir_content_names):
                fs = path.getsize(sensor_dir_content[i])
                if fs == 0:
                    delete_file(sensor_dir_content[i], reason='0-byte file', dry_run=dry_run)
                    continue
                if psbl_rawfile in ['SA_log.out', 'cronjobs_help.sh', 'cronlog.txt', 'cronlog.out', 'data_collect.py']:
                    filename_spec[i] = 0
                    continue
                if not (psbl_rawfile.endswith('.raw') & ('_' in psbl_rawfile)):
                    filename_spec[i] = -1
                    delete_file(sensor_dir_content[i], reason='content of sensor dir is not a raw data or log file', dry_run=dry_run)
                    continue                    
                rawfile_split = psbl_rawfile.replace('.raw', '').split('_')
                if len(rawfile_split) == 2:
                    # this is an 'old old' file type
                    filename_spec[i] = 1
                    this_file_dt = dt.strptime(psbl_rawfile, '%Y%m%d%H%M%S_%f.raw')
                    this_file_relay = None # TODO: figure this out from the SA_log.out or cronlog.txt or cronlog.out or data_collect.py, if available
                    # TODO: maybe read this info from some sort of user-provided csv and convert to a "2" or "3" spec filename automatically
                    all_lons[i] = np.nan
                    all_lats[i] = np.nan
                    all_alts[i] = np.nan
                    this_file_gps_err = 0 # if there is no GPS data, the error is defined to be 0
                    cpu_ids[i] = 0
                elif len(rawfile_split) == 3:
                    # this is an 'old' file type
                    filename_spec[i] = 2
                    this_file_dt = dt.strptime(rawfile_split[0]+rawfile_split[1], '%Y%m%d%H%M%S%f')
                    all_lons[i] = np.nan
                    all_lats[i] = np.nan
                    all_alts[i] = np.nan
                    this_file_gps_err = 0
                    cpu_ids[i] = 0
                    this_file_relay = rawfile_split[2]
                elif len(rawfile_split) == 9:
                    # this is a current filename
                    filename_spec[i] = 3
                    this_file_dt = dt.strptime(rawfile_split[0]+rawfile_split[1]+rawfile_split[2], '%Y%m%d%H%M%S%f')
                    all_lats[i] = np.nan if rawfile_split[3] == 'NO' else float(rawfile_split[3])
                    all_lons[i] = None if rawfile_split[4] == 'FIX' else float(rawfile_split[4])
                    all_alts[i] = None if rawfile_split[5] == '2Donly' else float(rawfile_split[5])
                    this_file_gps_err = float(rawfile_split[6]) if ~np.isnan(all_lons[i]) else 0
                    cpu_ids[i] = int(rawfile_split[7], 16)
                    this_file_relay = rawfile_split[8]
                elif len(rawfile_split) == 6:
                    # This is a current file with no GPS... see https://github.com/wx4stg/Bruning_Slow_Antenna_Software/issues/3
                    filename_spec[i] = 3
                    this_file_dt = dt.strptime(rawfile_split[0]+rawfile_split[1]+rawfile_split[2], '%Y%m%d%H%M%S%f')
                    all_lons[i] = np.nan
                    all_lats[i] = np.nan
                    all_alts[i] = np.nan
                    this_file_gps_err = float(rawfile_split[3]) if ~np.isnan(all_lons[i]) else 0
                    cpu_ids[i] = int(rawfile_split[4], 16)
                    this_file_relay = rawfile_split[5]
                    new_file_name = f'{this_file_dt.strftime("%Y%m%d_%H%M%S_%f")}_NO_FIX_2Donly_{this_file_gps_err:.2f}_{cpu_ids[i]:x}_{this_file_relay}.raw'
                    new_file_path = sensor_dir_content[i].replace(psbl_rawfile, new_file_name)
                    move_file(sensor_dir_content[i], new_file_path, reason='Fix for https://github.com/wx4stg/Bruning_Slow_Antenna_Software/issues/3', dry_run=dry_run)
                # Attempt to recover location if missing
                if np.any(np.isnan([all_lats[i], all_lons[i], all_alts[i]])):
                    no_fix_filename = f'{this_file_dt.strftime("%Y%m%d_%H%M%S_%f")}_NO_FIX_2Donly_{this_file_gps_err:.2f}_{cpu_ids[i]:x}_{this_file_relay}.raw'
                    no_fix_path = path.join(sensor_dir, no_fix_filename)
                    if path.exists(no_fix_path):
                        lons_to_avg = []
                        lats_to_avg = []
                        alts_to_avg = []
                        if i != 0:
                            lons_to_avg.append(all_lons[i-1])
                            lats_to_avg.append(all_lats[i-1])
                            alts_to_avg.append(all_alts[i-1])
                        if i != (len(sensor_dir_content)-1):
                            lons_to_avg.append(all_lons[i+1])
                            lats_to_avg.append(all_lats[i+1])
                            alts_to_avg.append(all_alts[i+1])
                        if len(lons_to_avg) > 0 and len(lats_to_avg) > 0 and len(alts_to_avg) > 0 and ~np.all(np.isnan(lons_to_avg)) and ~np.all(np.isnan(lats_to_avg)) and ~np.all(np.isnan(alts_to_avg)):
                            this_file_lon = np.mean(lons_to_avg)
                            this_file_lat = np.mean(lats_to_avg)
                            this_file_alt = np.mean(alts_to_avg)
                            if np.all([~np.isnan(this_file_lon), ~np.isnan(this_file_lat), ~np.isnan(this_file_alt)]):
                                new_file_name = f'{this_file_dt.strftime("%Y%m%d_%H%M%S_%f")}_{this_file_lat:.3f}_{this_file_lon:.3f}_{this_file_alt:.1f}_{this_file_gps_err:.2f}_{cpu_ids[i]:x}_{this_file_relay}.raw'
                                new_file_path = sensor_dir_content[i].replace(psbl_rawfile, new_file_name)
                                move_file(no_fix_path, new_file_path, reason='recovered GPS location from neighboring files', dry_run=dry_run)
                # Detect misplaced rawfile dates
                if this_file_dt.replace(hour=0, minute=0, second=0, microsecond=0) != this_date.replace(hour=0,	minute=0, second=0, microsecond=0):
                    new_file_path = path.join(archive_root, this_file_dt.strftime('%Y%m%d'), path.basename(sensor_dir), psbl_rawfile)
                    move_file(sensor_dir_content[i], new_file_path, reason='file has different date than parent directory', dry_run=dry_run)
                dts_this_day.append(this_file_dt)
                lons_this_day.append(all_lons[i])
                lats_this_day.append(all_lats[i])
                paths_this_day.append(sensor_dir_content[i])
            # Detect different file specs in same directory
            filename_spec_valid = filename_spec.copy()[(filename_spec != 0) & (filename_spec != -1)] 
            if filename_spec_valid.size > 0:
                if not np.all(filename_spec_valid == filename_spec_valid[0]):
                    unhandleable_file(f'{sensor_dir} contains raw files generated by different OS revisions, I\'m not sure how to proceed.', dry_run=dry_run)
            # Detect multiple CPU serials in same directory
            cpu_ids_valid = cpu_ids.copy()[cpu_ids != 0]
            if cpu_ids_valid.size > 0:
                unique_cpu_ids = np.unique(cpu_ids_valid)
                if unique_cpu_ids.size == 1:
                    this_date_ids[str(sensor_num).zfill(2)] = unique_cpu_ids[0]
                else:
                    ids_found = ', '.join([f'0x{id:x}' for id in unique_cpu_ids])
                    this_date_ids[str(sensor_num).zfill(2)] = ids_found
                    unhandleable_file(f'{sensor_dir} contains raw files generated by CPU serial numbers: {ids_found}, I\'m not sure how to proceed.', dry_run=dry_run)
        # Log CPU serial numbers if requested
        if args.cpu_serial_log is not None:
        # Detect same CPU serial in different sensor directories at same time
            for sensor_to_log, dserial_to_log in this_date_ids.items():
                if type(dserial_to_log) is str:
                    serials = dserial_to_log.split(',')
                    serials_to_log = [f'{str(id.strip())}' for id in serials]
                else:
                    serials_to_log = [f'0x{dserial_to_log:x}']
                for serial_to_log in serials_to_log:
                    if serial_to_log not in cpu_df.index:
                        cpu_df.loc[serial_to_log] = np.full(cpu_df.columns.size, 0, dtype=int)
                    cpu_df.at[serial_to_log, this_date.strftime('%Y%m%d')] = int(sensor_to_log)
        if len(this_date_ids.keys()) > 1:
            this_date_ids_values = []
            for c in this_date_ids.values():
                if type(c) is int:
                    this_date_ids_values.append(c)
                elif type(c) is str:
                    for part in c.split(','):
                        this_date_ids_values.append(int(part.strip(), 16))
            unique_ids_this_day, id_counts_this_day = np.unique(this_date_ids_values, return_counts=True)
            if np.any(id_counts_this_day > 1):
                duplicate_ids_this_day = unique_ids_this_day[id_counts_this_day > 1]
                for dup_id in duplicate_ids_this_day:
                    sensors_with_dup_id = [f'{date_dir}/sensor_{sensor}/' for sensor, cpu_id in this_date_ids.items() if cpu_id == dup_id]
                    sensors_with_dup_id_str = ', '.join(sensors_with_dup_id)
                    unhandleable_file(f'Same CPU serial 0x{dup_id:x} in sensor directories: {sensors_with_dup_id_str}', dry_run=dry_run)
        dts_this_day = np.array(dts_this_day)
        lons_this_day = np.array(lons_this_day)
        lats_this_day = np.array(lats_this_day)
        paths_this_day = np.array(paths_this_day)
        # Prune uninteresting data by LMA
        if args.lma_data is not None and len(paths_this_day) > 0 and np.all(~np.isnan(lons_this_day) & ~np.isnan(lats_this_day)):
            first_time_today = np.min(dts_this_day)
            first_time_today = first_time_today.replace(minute=(first_time_today.minute // 10) * 10, second=0, microsecond=0)
            last_time_today = np.max(dts_this_day)
            last_time_today = last_time_today.replace(minute=(last_time_today.minute // 10) * 10, second=0, microsecond=0) + timedelta(minutes=10)
            dts_this_day = dts_this_day.astype('datetime64[s]')
            # Find LMA files that cover this time range
            lma_pattern = path.join(args.lma_data, this_date.strftime('%Y/%b/%d/*_%y%m%d_*_0600.dat.flash.h5'))
            lma_file_paths = sorted(glob(lma_pattern))
            tomorrow = this_date + timedelta(days=1)
            lma_file_paths += sorted(glob(path.join(args.lma_data, tomorrow.strftime('%Y/%b/%d/*_%y%m%d_000000_0600.dat.flash.h5'))))
            try:
                lma_file_times = np.array([dt.strptime(''.join(path.basename(lf).replace('_2011_', '_').split('_')[1:3]), '%y%m%d%H%M%S') for lf in lma_file_paths])
            except Exception as e:
                print(f'Error interpreting LMA file times from pattern {lma_pattern}: {e}')
                print(lma_pattern)
                raise e
            files_to_read_mask = (lma_file_times >= first_time_today) & (lma_file_times <= last_time_today)
            lma_file_times = lma_file_times[files_to_read_mask]
            lma_file_paths = np.array(lma_file_paths)[files_to_read_mask].tolist()
            flash_df = pd.DataFrame()
            for i, this_path in enumerate(lma_file_paths):
                ds = h5py.File(this_path, 'r')
                this_flash_df = pd.DataFrame(ds['flashes'][lma_file_times[i].strftime('LMA_%y%m%d_%H%M%S_600')][:])
                this_flash_df['flash_dt'] = pd.Timestamp(lma_file_times[i].replace(hour=0, minute=0, second=0, microsecond=0)) + pd.to_timedelta(this_flash_df['start'], unit='s')
                flash_df = pd.concat([flash_df, this_flash_df], ignore_index=True).reset_index(drop=True)
                ds.close()
            sensor_X, sensor_Y, sensor_Z = geosys.toECEF(lons_this_day, lats_this_day, np.zeros(lons_this_day.shape))
            flash_X, flash_Y, flash_Z = geosys.toECEF(flash_df['ctr_lon'].values, flash_df['ctr_lat'].values, np.zeros(flash_df['ctr_lon'].shape))
            distances = ((sensor_X.reshape((-1, 1)) - flash_X.reshape((1, -1)))**2
                       + (sensor_Y.reshape((-1, 1)) - flash_Y.reshape((1, -1)))**2
                       + (sensor_Z.reshape((-1, 1)) - flash_Z.reshape((1, -1)))**2)**0.5
            distances_thresholded = distances <= 100e3 # 100 km
            times_differences = np.abs((dts_this_day.reshape((-1, 1)) - flash_df['flash_dt'].values.reshape((1, -1))).astype('timedelta64[s]').astype(float))
            times_differences_thresholded = times_differences <= 1800 # 30 minutes
            flashes_nearby = np.any(distances_thresholded & times_differences_thresholded, axis=1)
            paths_to_rm = paths_this_day[~flashes_nearby]
            for pr in paths_to_rm:
                delete_file(pr, reason='no nearby LMA flashes within 100 km and 30 minutes', dry_run=dry_run)
        new_date_dir_content = sorted(glob(path.join(date_dir, '*')))
        for sensor_dir in sensor_dirs:
            new_sensor_dir_content = sorted(glob(path.join(sensor_dir, '*')))
            if len(new_sensor_dir_content) == 0:
                delete_file(sensor_dir, reason='empty sensor directory', dry_run=dry_run)
        if len(new_date_dir_content) == 0:
            delete_file(date_dir, reason='empty date directory', dry_run=dry_run)
    if args.cpu_serial_log is not None:
        cpu_df.to_csv(args.cpu_serial_log, index_label='CPU Serial')

    # TODO: Find dates of old deployments and associate lat/lon and maybe even CPU ID to convert to new file format
    # TODO: plot time differences between files to find gaps and misplaced files
