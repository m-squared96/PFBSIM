#!/usr/bin/python3

import os

import numpy as np
import pandas as pd

#TODO: Handle generic wave data generation

def signal_parent(time_array,complexity,noise_strength,method,sampling_frequency,fmin,fmax):
    if method in ('wave','mkid'):
        signal = noise_strength*np.random.random(len(time_array))

        if method == 'wave':
            signal,filename = wave_gen(signal,time_array,complexity,sampling_frequency)

            return signal,filename

        elif method == 'mkid':
            signal,filename = mkid_gen(signal,time_array,complexity,sampling_frequency,False,fmin,fmax)
            signal_p,filename_p = mkid_gen(signal,time_array,complexity,sampling_frequency,True,fmin,fmax)
            
            return signal,filename,signal_p,filename_p

    else:
        raise ValueError('"method" parameter should be set to either "wave" or "mkid"')

def wave_gen(signal,time_array,complexity,sampling_frequency):
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
    
    return signal,filename

def mkid_gen(signal,time_array,complexity,sampling_frequency,perturbation,fmin,fmax):
    min_freq = fmin
    max_freq = fmax
    freq_spacing = (max_freq-min_freq)/(complexity-1)

    if freq_spacing < 2e6:
        raise ValueError("Too many resonators: adjust bandwidth to accomodate")

    elif freq_spacing >= 2e6:
        lut_gen(fmin,fmax,freq_spacing,complexity)

        if not perturbation:
            res_freq = min_freq
            res_count = 1

            while res_count <= complexity:
                signal += np.sin(res_freq*2*np.pi*time_array)
                
                res_count += 1
                res_freq += freq_spacing

        elif perturbation:
            delta = np.random.uniform(low=-1.0,high=1.0,size=complexity)
            res_freq = min_freq
            res_count = 1

            while res_count <= complexity:
                res_measured = res_freq + delta[int(complexity-1)]*1e6
                signal += np.sin(res_measured*2*np.pi*time_array)

                res_count += 1
                res_freq += freq_spacing

        filename = 'mkid_' + str(complexity) + '_' + str(min_freq/1e9) + '_' + str(max_freq/1e9)
        if perturbation:
            filename += '_P'
    
    return signal,filename

def lut_gen(fmin,fmax,spacing,res_count):
    lut_filename = "LUTs/LUT_" + str(res_count) + "_" + str(fmin/1e9) + "_" + str(fmax/1e9) + ".npy"
    if not(os.path.isfile(lut_filename)):
        print("\nGenerating LUT file:")
        print("Min. Frequency: " + str(fmin/1e9) + "GHz")
        print("Max. Frequency: " + str(fmax/1e9) + "GHz")
        print("Number of Resonators: " + str(res_count))
        print("Location: " + lut_filename)
        f = fmin
        flist = []

        while f <= fmax:
            flist.append(f)
            f += spacing

        flist = np.array(flist,float)
        np.save(lut_filename,flist)

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
        print("Creating Data/ subdirectory")
        os.system("mkdir Data")

    if not(os.path.isdir("LUTs")):
        print("Creating LUTs/ subdirectory")
        os.system("mkdir LUTs")

if __name__ == '__main__':

    directory_check()

    Npoint = 4096
    multiple = 1000
    complexity = 20 # Number of signals/resonators
    noise_strength = 0.5

    # Define min and max resonator frequencies for use in MKID file generation
    min_freq = 2.0e9
    max_freq = 4.0e9

    fs = 1e10 # Sampling frequency
    time_length = (Npoint*multiple)/fs
    time = np.arange(0,time_length,1/fs)

    method = input('Enter signal type (wave/mkid):  ')
    print('Generating files of type', method)

    if method == 'mkid':
        signal,filename,signal_p,filename_p = signal_parent(time,complexity,noise_strength,method,fs,min_freq,max_freq)
        sigframe,sigframe_p = pd.DataFrame(columns=['Time','Signal']),pd.DataFrame(columns=['Time','Signal'])
        
        print('\nData files:')

        for name,sig,frame in zip((filename,filename_p),(signal,signal_p),(sigframe,sigframe_p)):
            name += "_N" + str(noise_strength)
            frame['Time'] = time
            frame['Signal'] = sig

            print('Location: Data/' + name + ".csv")
            print('Size:', str(frame.shape))

            frame.to_csv('Data/' + name + '.csv', index=False)

        print("\nComplete.")

    elif method =='wave':

        print("Wave file generation under development")