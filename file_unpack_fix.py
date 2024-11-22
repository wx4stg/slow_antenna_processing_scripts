from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import struct
import datetime
import sys

import scipy.signal as signal


def convert_adc_to_decimal(value):
    modulo = 1 << 24
    max_value = (1 << 23) - 1
    if value > max_value:
        value -= modulo
    return value

def decode_data_packet(mp):

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

    #print(mp[3:4])
    #print(adc_ba)
    adc_reading = struct.unpack('>i', adc_ba[:])[0]

    adc_reading = mp[1]
    adc_reading = (adc_reading << 8) | mp[2]
    adc_reading = (adc_reading << 8) | mp[3]
    adc_reading = convert_adc_to_decimal(adc_reading)


    result['adc_reading'] = adc_reading
    return result

SERIAL_SPEED = 2000000



#pin_relay_a = 5
#pin_relay_b = 6
#pin_relay_c = 13
#pin_overflow_buffer = 10
#pin_overflow_serial = 11

#GPIO.setmode(GPIO.BCM)
#GPIO.setup(pin_relay_a, GPIO.OUT)
#GPIO.setup(pin_relay_b, GPIO.OUT)
#GPIO.setup(pin_relay_c, GPIO.OUT)
#GPIO.setup(pin_overflow_buffer, GPIO.IN)
#GPIO.setup(pin_overflow_serial, GPIO.IN)


#ba = bytearray(do_run())

#print(do_run())

#GPIO.output(pin_relay_a, GPIO.LOW)
#GPIO.output(pin_relay_b, GPIO.LOW)
#GPIO.output(pin_relay_c, GPIO.HIGH)
sleep(2)
#ba = do_run()
filename = sys.argv[1] #'20220807061848.txt'
with open(filename, mode = 'rb') as file:
    ba = file.read()
#ba = bytearray(ba)
#print(ba)
print(type(ba))

data_start_bytes = []
data_packet_length = 8
# Determine the valid starting bytes for data packets
for i in range(len(ba) - data_packet_length):
    if (ba[i] == 190) and (ba[i+data_packet_length] == 239):
        data_start_bytes.append(i)

# data_raw_packets = []
# data_packets = []
# this_packet_length = 12
this_packet_length = data_packet_length + 1
# for sb in data_start_bytes[:-1]:
    # data_raw_packets.append(ba[sb:sb+this_packet_length])
    # data_packets.append(decode_data_packet(ba[sb:sb+this_packet_length]))
data_raw_packets = [ba[sb:sb+this_packet_length] for sb in data_start_bytes[:-1]]
data_packets = [decode_data_packet(b) for b in data_raw_packets]


def notch_sixty(s, fs):
    f0 = 60.0  # Frequency to be removed from signal (Hz)
    Q = 2.0  # Quality factor = center / 3dB bandwidth
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, s)

# starts = []
# adc_ready = []
# adc = []
# end = []
# for dp in data_packets:
#     starts.append(dp['start_byte'])
#     adc_ready.append(dp['adc_pps_micros'])
#     adc.append(dp['adc_reading'])
#     end.append(dp['end_byte'])
starts = [dp['start_byte'] for dp in data_packets]
adc_ready = [dp['adc_pps_micros'] for dp in data_packets]
adc = [dp['adc_reading'] for dp in data_packets]
end = [dp['end_byte'] for dp in data_packets]

starts = np.array(starts)
adc_ready = signal.medfilt(np.array(adc_ready), 7)
adc = signal.medfilt(np.array(adc), 7)
end = np.array(end)

print(adc.shape, adc.dtype)
delta_t_adc = (adc_ready[-1]-adc_ready[0])*1e-6
# sample_rate = adc_ready.shape[0]/delta_t_adc
sample_rate = 1.0e6/np.median(np.ediff1d(adc_ready))
print(f"Elapsed time {delta_t_adc:6.3} s with sample rate {sample_rate:6.1f} Hz")

t = np.arange(adc.shape[0])/sample_rate
adc_filt = adc#notch_sixty(adc, sample_rate)


fig, axs = plt.subplots(2,2, sharex=True, figsize=(10, 6.5))
fig.suptitle(filename)

axs[0,0].specgram(adc-adc.mean(), Fs=sample_rate)#, window=np.blackman(256))
axs[0,0].set_title('Spectrogram (raw)')
axs[0,0].set_ylabel('Frequency (Hz)')
axs[0,0].set_ylim(0, 360)

axs[1,0].specgram(adc_filt-adc_filt.mean(), Fs=sample_rate, window=np.blackman(256))
axs[1,0].set_title('Spectrogram (filtered)')
axs[1,0].set_ylabel('Frequency (Hz)')
axs[1,0].set_ylim(0, 360)
axs[1,0].set_xlabel('Time (s)')
# axs[0,0].plot(t, starts)
# axs[0,0].set_title('Starts')

# axs[1,0].plot(end)
# axs[1,0].set_title('Ends')

axs[0,1].plot(t, adc_ready-adc_ready[0])
axs[0,1].set_title('ADC ready microsecond')

bits_to_volts = (5/((2**24)-1))
axs[1,1].plot(t, adc_filt*bits_to_volts)#-adc_filt.mean())
axs[1,1].set_title('ADC (filtered)')
axs[1,1].set_xlabel('Time (s)')
axs[1,1].set_ylim([-0.6,2.6])



#plt.plot(adc)
plt.show()
name = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.txt'
print(name)
#File1 = open(name, 'w')
data = np.column_stack([adc, adc_ready, starts, end])
#np.savetxt(name, data, fmt=['%d','%d', '%d','%d'])
#print(type(adc))

#L = ['THIS IS A TEST \n','THIS IS ANOTHER TEST']
#s = "hello\n"
#File1.writelines(L)
#File1.write(s)
#File1.write(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
#File1.close()
print('done with file')

#GPIO.output(pin_relay_a, GPIO.LOW)
#GPIO.output(pin_relay_b, GPIO.LOW)
#GPIO.output(pin_relay_c, GPIO.LOW)

#GPIO.cleanup()


