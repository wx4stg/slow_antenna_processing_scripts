#!/usr/bin/env python3

import struct
import numpy as np

def read_SA_file(file_to_read, packet_length=9, previous_file=None):
    with open(file_to_read, 'rb') as f:
        ba = f.read()
    this_bytes = np.frombuffer(ba, dtype=np.uint8)
    for j, byte in enumerate(this_bytes):
        # Find the first byte that is the header of the an ADC packet
        # The ADC packet has a header of 'BE' and a footer of 'EF' (190 = BE, 239 = EF)
        # So look for a footer immediately followed by a header,
        # followed by another footer/header pair 9 bytes later
        if byte == 190 and this_bytes[j+packet_length-1] == 239 and this_bytes[j+packet_length] == 190:
            length_to_include = ((this_bytes[j:].shape[0]) // packet_length) * packet_length
            adc_packets = this_bytes[j:length_to_include+j].reshape(-1, packet_length)
            # Now check all of the first and last columns to make sure they are all headers and footers respectively
            if np.all(adc_packets[:, 0] == 190) and np.all(adc_packets[:, -1] == 239):
                break
        if j >= packet_length:
            raise ValueError('Could not find first ADC packet in file.')
    if previous_file and j > 0:
        with open(previous_file, 'rb') as f:
            f.seek(-packet_length+j, 2)
            last_ba = f.read()
        last_bytes = np.frombuffer(last_ba, dtype=np.uint8)
        first_packet_recovered = np.append(last_bytes, this_bytes[:j])
        first_packet_recovered = first_packet_recovered.reshape(-1, packet_length)
        if first_packet_recovered[0, 0] == 190 and first_packet_recovered[0, -1] == 239:
            adc_packets = np.append(first_packet_recovered, adc_packets, axis=0)
        else:
            print('First packet not recovered')
    return adc_packets

def decode_SA_array(data_array):
    # Extract the ADC local clock and convert the hexadecimal value to base 10
    # ADC clock is sent as 4 bytes, least significant byte first, so multiply by 256^0, 256^1, 256^2, 256^3
    adc_pps_micros = np.sum(data_array[:, 4:8] * (256 ** np.arange(4)), axis=1)
    # Extract the ADC reading and convert to decimal.
    # The ADC reading is sent as 3 bytes, most significant byte first, so multiply by 256^2, 256^1, 256^0
    adc_reading_dec = np.sum(data_array[:, 1:4] * np.flip(256 ** np.arange(3)), axis=1)
    # The ADC reading is sent as a signed 24-bit integer, so we need to convert it to a signed integer
    # If the ADC reading is greater than 2^23 - 1, then it is a negative number
    adc_reading_overflow_mask = adc_reading_dec > (2**23 - 1)
    adc_reading = adc_reading_dec - adc_reading_overflow_mask.astype(int) * (2**24)
    
    return adc_pps_micros, adc_reading

def convert_adc_to_decimal(value):
    modulo = 1 << 24
    max_value = (1 << 23) - 1
    if value > max_value:
        value -= modulo
    return value


def old_readSAfile(filepath):
    data_raw_packets=[]
    data_start_bytes = []
    data_packet_length = 8
    data_packets = []
    this_packet_length = data_packet_length + 1
    with open(filepath, mode = 'rb') as file:
        ba = file.read()
    for i in range(len(ba) - data_packet_length):
        if (ba[i] == 190) and (ba[i+data_packet_length] == 239):
            data_start_bytes.append(i)
    data_raw_packets.extend([ba[sb:sb+this_packet_length] for sb in data_start_bytes[:-1]])
    data_packets = [decode_data_packet(b) for b in data_raw_packets]
    return data_packets    
    
def decode_data_packet(mp):
    result = dict()
    result['start_byte'] = struct.unpack('B', mp[0:1])[0]
    result['b1'] = struct.unpack('B', mp[1:2])[0]
    result['b2'] = struct.unpack('B', mp[2:3])[0]
    result['b3'] = struct.unpack('B', mp[3:4])[0]
    result['adc_pps_micros'] = struct.unpack('I', mp[4:8])[0]
    result['end_byte'] = struct.unpack('B', mp[8:9])[0]
    adc_ba = bytearray()
    adc_ba += mp[1:2]
    adc_ba += mp[2:3]
    adc_ba += mp[3:4]
    adc_ba += b'\x00'

    adc_reading = struct.unpack('>i', adc_ba[:])[0]

    adc_reading = mp[1]
    adc_reading = (adc_reading << 8) | mp[2]
    adc_reading = (adc_reading << 8) | mp[3]
    adc_reading = convert_adc_to_decimal(adc_reading)

    result['adc_reading'] = adc_reading
    return result