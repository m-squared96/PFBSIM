#!/usr/bin/python3

import os
import glob
import readline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import toolkit as pfb

#TODO: Split FFTs

class FilenameCompleter:

    def __init__(self,filenames):

        self.options = sorted(filenames)

    def complete(self,text,state):

        if state == 0: # Generate possible matches on first trigger
            if text: # Cache entries that start with entered text
                matches = []
                for f in self.options:
                    if f.startswith(text):
                        matches.append(f)

                self.matches = matches

            else: # No text entered --> all matches possible
                self.matches = self.options[:]

        # Return match indexed by state variable
        try:
            return self.matches[state]
        except IndexError:
            return None

def readfile(N):

    if os.path.isdir("Data"): # Checks if Data/ subdirectory exists

        file_list = glob.glob('Data/*.csv') # Returns a list of CSV files in Data/

        if len(file_list) > 0:
            
            print("Files in 'Data/' repository:")
            print("\n")

            filenames = []
            for f in file_list:
                filenames.append(f[5:-4])
                print(f[5:-4])
            
            print("\n")

            file_completer = FilenameCompleter(filenames)
            readline.set_completer(file_completer.complete)
            readline.parse_and_bind('tab: complete')

            filename = input("Enter data filename:  ")
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

            return signal, file_length, filename

        else:

            raise ValueError("No files in 'Data/' repository")

    else:

        raise ValueError("No 'Data/' directory exists.")

def pfb_handler(signal,Npoint,file_length,window,lo,lut):

    PFB = pfb.FilterBank(signal,Npoint,file_length,window,lo,lut)
    freqs = np.fft.fftfreq(n=PFB.N)*PFB.fs

    plt.figure()
    plt.plot(freqs,np.abs(PFB.I_fft + PFB.Q_fft))
    #plt.plot(freqs,np.abs(mixed_fft),label="Mixed")
    #plt.legend()

def fft_handler(signal,Npoint,file_length,lo):

    tool = pfb.FFTGeneric(signal,Npoint,file_length,lo)
    freqs = np.fft.fftfreq(n=tool.N)*tool.fs

    # plt.figure()
    # plt.plot(freqs,np.abs(fft))

def main():

    N = 4096 # Point spec for the FFT
    mixing_lo = 2.0e9 # Local oscillator frequency for the IQ mixer

    min_freq = 2.0e9
    max_freq = 4.0e9

    signal, file_length, filename = readfile(N)

    mode = input("Enter operational mode (pfb/fft):   ")
    
    if mode == 'pfb':
        num_str = filename[5:7]
        fmin_str = filename[8:11]
        fmax_str = filename[12:15]
        lut_filename = "LUTs/LUT_" + num_str + "_" + fmin_str + "_" + fmax_str + ".npy"

        print("Attempting to load LUT file: " + lut_filename)

        try:
            lut = np.load(lut_filename)
            print("Loaded LUT file successfully")

        except FileNotFoundError:
            lut = None
            print("LUT file not loaded, fine channelisation cannot occur")

        pfb_handler(signal,N,file_length,'hamming',mixing_lo,lut)        

    elif mode == 'fft':
        fft_handler(signal,N,file_length,mixing_lo)        

main()
plt.show()