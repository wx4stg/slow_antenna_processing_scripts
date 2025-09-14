#!/usr/bin/env python3

import argparse
from os import path, walk, listdir, remove, rmdir

parser = argparse.ArgumentParser(description='Slow antenna archive pruning and sorting.')

parser.add_argument('--archive-root', '-r', type=str, required=True, default='/mnt/reservoir/SA_DATA/', help='Top level directory of slow antenna data archive.')
parser.add_argument('--lma-data', '-l', type=str, default=None, help='Directory containing LMA data for correlation with slow antenna data. Leave unspecified to skip pruning.')
parser.add_argument('--dry-run', '-n', action='store_true', help='Log actions without making any changes.')

if __name__ == '__main__':
    args = parser.parse_args()

    archive_root = args.archive_root
    dry_run = args.dry_run

    for dirpath, dirnames, filenames in walk(archive_root):
        print(f'Processing directory: {dirpath}')
        print('=======================')

        # TODO: Check for dates misaligned with parent directory and move to correct dir
        # TODO: Check for directories that aren't dates and arent sensor_XX
        # TODO: check for directories containing multiple CPU IDs
        # TODO: find same CPU ID in different sensor directories at same time
        # TODO: Create a log of CPU IDs and sensor numbers to move sensor IDs to the correct locations
        # TODO: Find dates of old deployments and associate lat/lon and maybe even CPU ID to convert to new file format
        # TODO: find files with [this issue](https://github.com/wx4stg/Bruning_Slow_Antenna_Software/issues/3) and try to recover the missing locations?
        # TODO: Do the same for files with NO_FIX
        # TODO: find 0-byte files and remove them
        # TODO: plot time differences between files to find gaps and misplaced files
        # TODO: incorporate LMA based pruning of unininteresting data
        # TODO: remove empty directories
