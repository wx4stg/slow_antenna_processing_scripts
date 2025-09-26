#!/usr/bin/env python3
# Sorts TTU Slow Antenna network data and prunes unininteresting data by LMA
# Created 14 September 2025 by Sam Gardner <samuel.gardner@ttu.edu> 

import argparse
from os import path
from glob import glob
from datetime import datetime as dt
import numpy as np

parser = argparse.ArgumentParser(description='Slow antenna archive pruning and sorting.')
parser.add_argument('--archive-root', '-r', type=str, required=True, default='/mnt/reservoir/SA_DATA/', help='Top level directory of slow antenna data archive.')
parser.add_argument('--lma-data', '-l', type=str, default=None, help='Directory containing LMA data for correlation with slow antenna data. Leave unspecified to skip pruning.')
parser.add_argument('--dry-run', '-n', action='store_true', help='Log actions without making any changes.')


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
    date_directories = sorted(glob(path.join(archive_root, "*")))
    dates_to_proc = []
    for date_dir in date_directories:
        try:
            this_date = dt.strptime(path.basename(date_dir), '%Y%m%d')
        except ValueError as e:
            if 'does not match format' in str(e):
                delete_file(date_dir, reason='first level directory does not conform to date format string YYYYMMDD/', dry_run=dry_run)
        sensor_dirs = sorted(glob(path.join(date_dir, '*')))
        this_date_ids = dict()
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
            for i, psbl_rawfile in enumerate(sensor_dir_content_names):
                fs = path.getsize(sensor_dir_content[i])
                if fs == 0:
                    pass
                    # remove file
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
                    this_file_lat = None
                    this_file_lon = None
                    this_file_alt = None
                    this_file_gps_err = None
                    cpu_ids[i] = 0
                elif len(rawfile_split) == 3:
                    # this is an 'old' file type
                    filename_spec[i] = 2
                    this_file_dt = dt.strptime(rawfile_split[0]+rawfile_split[1], '%Y%m%d%H%M%S%f')
                    this_file_lat = None
                    this_file_lon = None
                    this_file_alt = None
                    this_file_gps_err = None
                    cpu_ids[i] = 0
                    this_file_relay = rawfile_split[2]
                elif len(rawfile_split) == 9:
                    # this is a current filename
                    filename_spec[i] = 3
                    this_file_dt = dt.strptime(rawfile_split[0]+rawfile_split[1]+rawfile_split[2], '%Y%m%d%H%M%S%f')
                    this_file_lat = None if rawfile_split[3] == 'NO' else float(rawfile_split[3])
                    this_file_lon = None if rawfile_split[4] == 'FIX' else float(rawfile_split[4])
                    this_file_alt = None if rawfile_split[5] == '2Donly' else float(rawfile_split[5])
                    this_file_gps_err = float(rawfile_split[6])
                    cpu_ids[i] = int(rawfile_split[7], 16)
                    this_file_relay = rawfile_split[8]
                elif len(rawfile_split) == 6:
                    # This is a current file with no GPS... see https://github.com/wx4stg/Bruning_Slow_Antenna_Software/issues/3
                    filename_spec[i] = 3
                    this_file_dt = dt.strptime(rawfile_split[0]+rawfile_split[1]+rawfile_split[2], '%Y%m%d%H%M%S%f')
                    this_file_lat = None
                    this_file_lon = None
                    this_file_alt = None
                    this_file_gps_err = float(rawfile_split[3])
                    cpu_ids[i] = int(rawfile_split[4], 16)
                    this_file_relay = rawfile_split[5]
                    new_file_name = f'{this_file_dt.strftime("%Y%m%d_%H%M%S_%f")}_NO_FIX_2Donly_{this_file_gps_err}_{cpu_ids[i]:x}_{this_file_relay}.raw'
                    new_file_path = sensor_dir_content[i].replace(psbl_rawfile, new_file_name)
                    move_file(sensor_dir_content[i], new_file_path, reason='Fix for https://github.com/wx4stg/Bruning_Slow_Antenna_Software/issues/3', dry_run=dry_run)
                # Detect misplaced rawfile dates
                if this_file_dt.replace(hour=0, minute=0, second=0, microsecond=0) != this_date.replace(hour=0,	minute=0, second=0, microsecond=0):
                    new_file_path = path.join(archive_root, this_file_dt.strftime('%Y%m%d'), path.basename(sensor_dir), psbl_rawfile)
                    move_file(sensor_dir_content[i], new_file_path, reason='file has different date than parent directory', dry_run=dry_run)
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
                    unhandleable_file(f'{sensor_dir} contains raw files generated by CPU serial numbers: {ids_found}, I\'m not sure how to proceed.', dry_run=dry_run)
        # Detect same CPU serial in different sensor directories at same time
        if len(this_date_ids.keys()) > 1:
            unique_ids_this_day, id_counts_this_day = np.unique(list(this_date_ids.values()), return_counts=True)
            if np.any(id_counts_this_day > 1):
                duplicate_ids_this_day = unique_ids_this_day[id_counts_this_day > 1]
                for dup_id in duplicate_ids_this_day:
                    sensors_with_dup_id = [f'{date_dir}/sensor_{sensor}/' for sensor, cpu_id in this_date_ids.items() if cpu_id == dup_id]
                    sensors_with_dup_id_str = ', '.join(sensors_with_dup_id)
                    unhandleable_file(f'Duplicate CPU serial 0x{dup_id:x} in sensor directories: {sensors_with_dup_id_str}', dry_run=dry_run)


    # TODO: Create a log of CPU IDs and sensor numbers to move sensor IDs to the correct locations
    # TODO: Find dates of old deployments and associate lat/lon and maybe even CPU ID to convert to new file format
    # TODO: Do the same for files with NO_FIX
    # TODO: find 0-byte files and remove them
    # TODO: plot time differences between files to find gaps and misplaced files
    # TODO: incorporate LMA based pruning of unininteresting data
    # TODO: remove empty directories
