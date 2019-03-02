#!/usr/bin/python3

import os
import glob
import readline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import toolkit as pfb

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

            return signal, file_length

        else:

            raise ValueError("No files in 'Data/' repository")

    else:

        raise ValueError("No 'Data/' directory exists.")

def main():

    N = 4096 # Point spec for the FFT
    mixing_lo = 1.5e9 # Local oscillator frequency for the IQ mixer

    signal, file_length = readfile(N)
    signal['Signal'] = pfb.signal_mixer(signal['Signal'],mixing_lo,signal['Time'])

    mode = input("Enter operational mode (pfb/fft):   ")
    
    if mode == 'pfb':

        PFB = pfb.FilterBank(signal,N,file_length,'hamming',1.5e9)
        pfb_branched = PFB.split()
        fft = PFB.fft(pfb_branched)
        freqs = PFB.frequencies()

        plt.figure()
        plt.plot(freqs,np.abs(fft),label="Unmixed")
        #plt.plot(freqs,np.abs(mixed_fft),label="Mixed")
        #plt.legend()
        plt.xlim(left=0)

    elif mode == 'fft':

        tool = pfb.FFTGeneric(signal,file_length,file_length)
        fft = tool.fft(tool.signal_array)
        freqs = tool.frequencies()

        plt.figure()
        plt.plot(freqs,np.abs(fft))

main()
plt.show()
