#!/usr/bin/env python3
# Simulates slow antenna ADC output from rawfile input
# Created 16 October 2025 by Sam Gardner <samuel.gardner@ttu.edu>

import serial
import sa_common
import argparse
import numpy as np


parser = argparse.ArgumentParser(description="Simulate slow antenna feather board serial output using a pi connected to itself.")
parser.add_argument('-i', '--input', nargs='+', required=True, help='Path to slow antenna raw files to transmit over the serial interface.')
parser.add_argument('-w', '--wait', action='store_true', help='Wait for interactive input before sending next packet.')
parser.add_argument('-s', '--serial-interface', type=str, default='/dev/ttyAMA0', help='Serial interface for sending raw data.')

if __name__ == '__main__':
    np.set_printoptions(formatter={'int':hex})
    args = parser.parse_args()
    ser = serial.Serial(args.serial_interface, 9600, timeout=1)
    for i, this_rawfile in enumerate(args.input):
        if i == 0:
            sa_data = sa_common.read_SA_file(this_rawfile)
        else:
            sa_data = sa_common.read_SA_file(this_rawfile, previous_file=args.input[i-1])
        n_packets = sa_data.shape[0]
        for packetnum in range(n_packets):
            this_packet = sa_data[packetnum, :]
            this_packet_str = b''.join([x.to_bytes(1, 'big') for x in this_packet.tolist()])
            print(f'Sending packet {packetnum}!')
            print(f'{this_packet}')
            ser.write(this_packet_str)
            print('-'*50)
            if args.wait:
                input('Press any key to continue...')
