#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#TODO: Figure out frequency array indexing

def main():

    filename = str(input("Enter the filename    "))
    filepath = "Data/" + filename + ".csv"
    signal = pd.read_csv(filepath)

    time_array = np.array(signal['Time'])
    signal_array = np.array(signal['Signal'])

    signal_fft = np.fft.fft(signal_array)

    n = signal_array.size
    time_int = timestep(time_array)

    xf_data = np.linspace(0.0,1.0/(4*time_int),int(n/4))

    plt.figure()
    plt.plot(xf_data,2.0/n * np.abs(signal_fft[:n//4]))
    plt.xlim(20,100)
    plt.show()

def timestep(time):
    
    interval1 = time[1] - time[0]
    interval2 = time[2] - time[0]

    if interval2 == 2*interval1:
        
        return interval1

    else:
        
        return (interval1 + 0.5*interval2)/2

main()
