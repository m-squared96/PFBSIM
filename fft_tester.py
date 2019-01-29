#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():

    filename = str(input("Enter the filename    "))
    filepath = "Data/" + filename + ".csv"
    signal = pd.read_csv(filepath)

    time_array = np.array(signal['Time'])
    signal_array = np.array(signal['Signal'])

    signal_fft = np.fft.rfft(signal_array)

    n = signal_array.size
    time_int = timestep(time_array)

    freqs = np.fft.rfftfreq(n,d=time_int)

    print(max(freqs))

    #plt.plot(freqs,signal_fft)
    #plt.show()

def timestep(time):
    
    interval1 = time[1] - time[0]
    interval2 = time[2] - time[0]

    if interval2 == 2*interval1:
        
        return interval1

    else:
        
        return (interval1 + 0.5*interval2)/2

main()
