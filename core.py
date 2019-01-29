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

        signal_pfb = pfb.pfb_init(signal,M,P,'hamming')

        plotter(signal_pfb)

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
