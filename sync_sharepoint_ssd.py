#!/usr/bin/env python3
# Transfer data from slow antenna and organize
# Created 16 August 2024 by Sam Gardner <samuel.gardner@ttu.edu>


from paramiko import SSHClient, AutoAddPolicy
from os import listdir, path, rename
from shutil import rmtree
import subprocess
from datetime import datetime as dt, UTC
from pathlib import Path
import pandas as pd
import numpy as np
from timeoutcontext import timeout
from alive_progress import alive_bar


kelcy_drive_path = '/media/lightning/Extreme_SSD/SA_SYNC/.'


def process_rsync_output(line, bar):
    line = line.decode()
    if 'rsync warning' in line:
        print(line)
    if '%' in line:
        pct = float(line.split()[1].replace('%', ''))/100
        bar(pct)
    elif 'receiving file list' in line:
        global rsync_started
        rsync_started = True
    elif 'speedup is' in line:
        bar(1)
    return -1


def run_rsync_with_progress(source_path, dest_path, should_SSH=True):
    rsync_cmd = ['rsync', '-uvr', '--no-i-r', '--info=progress2', source_path, dest_path]
    if should_SSH:
        prepend = ['sshpass', '-p', 'raspberry']
        prepend.extend(rsync_cmd)
        rsync_cmd = prepend
    
    rsync_task = subprocess.Popen(rsync_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    with alive_bar(manual=True, spinner='arrow', title='Transferring Data...', elapsed='total {elapsed}', stats='(ETA: {eta})') as bar:
        for line in iter(rsync_task.stdout.readline, b''):
            process_rsync_output(line, bar)

    rsync_task.stdout.close()
    rsync_task.wait()
    errors = rsync_task.stderr.read().decode()
    if errors:
        print('##################################################')
        print('WARNING! Errors detected in rsync process!')
        print('You may wish to CTRL+C to exit and try again before proceeding')
        print('')
        print(errors)
        print('')
        print('##################################################')
        input('Press RETURN to proceed, CTRL+C to exit...')
    return rsync_task


if __name__ == '__main__':
    input('Press RETURN when connected (or don\'t connect it to skip this)...')
    if path.exists(kelcy_drive_path):
        sync_to_ssd = input(f'Would you like to proceed with syncing data to {kelcy_drive_path}? [Y/n]: ')
        if sync_to_ssd.lower() == 'y' or sync_to_ssd == '':
            rsync_task = run_rsync_with_progress('/home/lightning/Desktop/ingest/.', kelcy_drive_path, should_SSH=False)
    should_sharepoint_upload = input('Would you like to upload the new data to SharePoint? [Y/n]: ')
    if should_sharepoint_upload.lower() == 'y' or should_sharepoint_upload == '':
        subprocess.Popen(['rclone', 'copy', '-P', kelcy_drive_path, 'ttu-ltg-sharepoint:/DATA_SYNC/'])
