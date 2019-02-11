#!/usr/bin/python3

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import toolkit as pfb

#TODO: Filename auto-completion
#TODO: Match frequency array with observed spikes

def main():

    N = 4096 # Point spec for the FFT

    signal, file_length = readfile(N)

    PFB = pfb.FilterBank(signal,N,file_length,'hamming')
    PFB.split()
    PFB.fft()
    PFB.frequencies()
    
    plotter(PFB)

def readfile(N):

    if os.path.isdir("Data"): # Checks if Data/ subdirectory exists

        file_list = glob.glob('Data/*.csv') # Returns a list of CSV files in Data/
        
        if len(file_list) > 0:
            
            print("Files in 'Data/' repository:")
            print("\n")

            for f in file_list:
                print(f[5:-4])
            
            print("\n")

            filename = str(input("Enter data filename:  "))
            signal = pd.read_csv("Data/" + filename + ".csv")
            
            file_length = len(signal['Signal'])

            # If file length is not a multiple of N, truncate the file (from the
            # end) to make it a multiple of N samples long.

            if file_length % N != 0:
                
                print('File length not a multiple of N (' + str(N) + ').')
                s_length = int(file_length/N)
                file_length = int(s_length*N)

                signal = signal[:file_length]
                print('File concatenated to length', str(file_length))

            return signal, file_length

        else:

            raise FileError("No files in 'Data/' repository")

    else:

        raise FileError("No 'Data/' directory exists.")

def plotter(PFB):
    
    plt.figure()
    plt.plot(PFB.freqs,np.abs(PFB.signal_f[:PFB.N//2]))

main()
plt.show()
