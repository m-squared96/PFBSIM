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

            return signal, file_length, filename

        else:
            raise FileNotFoundError("No files in 'Data/' repository")

    else:
        raise FileNotFoundError("No 'Data/' directory exists.")

def pfb_handler(signal,PFBdict,Sigdict,DSPdict):

    signal_array = signal['Signal']
    time_array = signal['Time']

    PFB = pfb.FilterBank(signal_array,time_array,PFBdict,Sigdict,DSPdict)

    #plt.figure()
    #plt.plot(PFB.freqs/1e9,pfb.dB(np.abs(PFB.fft)*1e6))
    #plt.title('PFB Signal Frequency Composition')
    #plt.xlabel('Frequency (GHz)')
    #plt.ylabel('Signal Strength (a.u.)')

def fft_handler(signal,N,Sigdict,DSPdict):

    signal_array = signal['Signal']
    time_array = signal['Time']

    tool = pfb.FFTGeneric(signal_array,time_array,N,Sigdict,DSPdict)

    plt.figure()
    plt.plot(tool.freqs,pfb.dB(np.abs(tool.fft)))
    plt.title('FFT Signal Frequency Composition')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Signal Strength (a.u.)')

def main():

    N = 4096 # Point spec for the FFT
    mixing = True
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
        'lpf':lpf_cutoff,
        'overlap':False,
        'ddc':False
    }

    mode = input("Enter operational mode (pfb/fft/both):   ")
    
    if mode == 'pfb':

        filename_components = tuple(filename.split('_'))
        num_str = filename_components[1]
        fmin_str = filename_components[2]
        fmax_str = filename_components[3]

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
        Sigdict['bandwidth'] = float(float(fmax_str) - float(fmin_str))*1e9
        Sigdict['fmax'] = float(fmax_str)*1e9
        Sigdict['fmin'] = float(fmin_str)*1e9

        pfb_handler(signal,PFBdict,Sigdict,DSPdict)

    elif mode == 'fft':
        fft_handler(signal,N,Sigdict,DSPdict)

    elif mode == 'both':
        fft_handler(signal,N,Sigdict,DSPdict)

        filename_components = tuple(filename.split('_'))
        num_str = filename_components[1]
        fmin_str = filename_components[2]
        fmax_str = filename_components[3]

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
        Sigdict['bandwidth'] = float(float(fmax_str) - float(fmin_str))*1e9
        Sigdict['fmax'] = float(fmax_str)*1e9
        Sigdict['fmin'] = float(fmin_str)*1e9

        pfb_handler(signal,PFBdict,Sigdict,DSPdict)

main()
plt.show()
