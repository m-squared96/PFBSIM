#!/usr/bin/python3

import os
import glob
import readline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import toolkit as pfb

#TODO: Signal reshaping/bin selection

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
            raise FileNotFoundError("No files in 'Data/' repository")

    else:
        raise FileNotFoundError("No 'Data/' directory exists.")

def pfb_handler(signal,PFBdict,Sigdict,DSPdict):

    PFB = pfb.FilterBank(signal,PFBdict,Sigdict,DSPdict)

    plt.figure()
    plt.plot(PFB.freqs,np.abs(PFB.I_fft + PFB.Q_fft))
    plt.xlim(0,PFB.fs/2)
    #plt.xlim(0,lpf_cutoff)

def fft_handler(signal,N,Sigdict,DSPdict):

    tool = pfb.FFTGeneric(signal,N,Sigdict,DSPdict)
    freqs = np.fft.fftfreq(n=tool.N)*tool.fs

    plt.figure()
    plt.plot(freqs,np.abs(tool.I_fft + tool.Q_fft))

def main():

    N = 2048 # Point spec for the FFT
    mixing = False
    mixing_lo = 2.0e9 # Local oscillator frequency for the IQ mixer
    lpf_cutoff = 2.1e9 # -3dB point of Butterworth IIR LPF
    taps = 4 # PFB taps
    window = 'hamming'

    signal, file_length, filename = readfile(N)

    Sigdict = {
        'length':file_length
    }

    DSPdict = {
        'lo':mixing_lo,
        'mixing':mixing,
        'lpf':lpf_cutoff
    }

    mode = input("Enter operational mode (pfb/fft):   ")
    
    if mode == 'pfb':
        num_str = filename[5:8]
        fmin_str = filename[9:12]
        fmax_str = filename[13:16]
        lut_filename = "LUTs/LUT_" + num_str + "_" + fmin_str + "_" + fmax_str + ".npy"

        print("\nAttempting to load LUT file: " + lut_filename)

        try:
            lut = np.load(lut_filename)
            print("Loaded LUT file successfully")

        except FileNotFoundError:
            lut = None
            print("LUT file not loaded, fine channelisation cannot occur")

        PFBdict = {
            'N':N,
            'taps':taps,
            'window':window
        }

        Sigdict['lut'] = lut
        Sigdict['complexity'] = int(num_str)

        pfb_handler(signal,PFBdict,Sigdict,DSPdict)

    elif mode == 'fft':
        fft_handler(signal,N,Sigdict,DSPdict)        

main()
plt.show()
