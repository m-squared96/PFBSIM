#!/usr/bin/python3

import os

import numpy as np
import pandas as pd

def signal_output(time_array,complexity,noise_strength,method,sampling_frequency):
    
    if method in ('wave','mkid'):
        signal = np.zeros(len(time_array))

        if noise_strength < 1.0:
            signal += noise_strength*np.random.random(len(signal))

        elif noise_strength >= 1.0:
            raise ValueError('Noise strength should be less than 1.0')

        if method == 'wave':
            
            coefficients = []
            coeff_count = 1
            while coeff_count <= complexity:
                coefficients.append(np.random.randint(1,9))
                coeff_count += 1
            coefficients = normalise(coefficients)

            frequencies = np.random.randint(low=1,high=5,size=complexity)*sampling_frequency/10
            filename = "wave_" + str(complexity)

            for f,c in zip(coefficients,frequencies):
                signal += c*np.sin(f*2*np.pi*time_array)
                filename += "_" + str(f)[:4]
        
        elif method == 'mkid':
            
            # Assuming a lower frequency limit of 2GHz, mixed down to ~ 0.2MHz
            min_freq = 2e5
            res_count = 1
            res_freq = min_freq

            while res_count <= complexity:
                signal += np.sin(res_freq*2*np.pi*time_array)
                res_count += 1
                res_freq += 2e3

            max_freq = res_freq

            filename = "mkid_" + str(complexity) + "_" + str(min_freq) + "_" + str(max_freq)

        return signal,filename

    else:
        raise ValueError('"method" parameter should be set to either "wave" or "mkid"')

def normalise(vector):
    '''
    Takes a vector and returns the normalised vector
    '''
    vector = np.array(vector)
    magnitude_s = 0
    for v in vector:
        magnitude_s += v**2

    magnitude = np.sqrt(magnitude_s)

    return vector*(1/magnitude)

def directory_check():

    '''
    Checks if 'Data/' subdirectory exists in current directory.
    If not, 'Data/' subdirectory is created.
    '''

    if not(os.path.isdir("Data")):
        os.system("mkdir Data")

def main():

    Npoint = 4096 
    multiple = 1000.789
    fs = 1e6 # Sampling frequency
    time_length = (Npoint*multiple)/fs
    time = np.arange(0,time_length,1/fs)
    
    method = 'mkid'
    signal, filename = signal_output(time,50,0.0,method,fs)

    print(signal.shape)
    print(filename)
    output_frame = pd.DataFrame({"Time":time, "Signal":signal})
    print("Signal data will be output to 'Data/' directory")

    directory_check()
    output_frame.to_csv("Data/" + filename + ".csv",index=False)

main()
