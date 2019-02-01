#!/usr/bin/python3

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import toolkit as pfb

#TODO: Figure out pfb output and how to isolate individual freqs

def main():

    M = 4
    P = 1024

    if os.path.isdir("Data"):

        filename = str(input("Enter data filename:  "))
        signal = pd.read_csv("Data/" + filename + ".csv")

        N_data = len(signal['Signal'])
        T_data = signal['Time'][1] - signal['Time'][0]

        signal_pfb = pfb.pfb_init(signal,M,P,'hamming')

        xf_data = np.linspace(0.0, 1.0/(4.0*T_data), int(N_data/4))

        plt.plot(xf_data,signal_pfb)
        plt.show()

    else:

        print("No 'Data/' directory exists.")

def plotter(processed):
    
    plt.figure()
    plt.title('Signal Power over Time')
    plt.plot(processed[0],np.abs(processed[1])**2)
    plt.xlabel('Channel')
    plt.ylabel('Power (dB)')

main()
plt.show()
