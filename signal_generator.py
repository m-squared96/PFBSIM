#!/usr/bin/python3

import os

import numpy as np
import pandas as pd

def signal_output(time_array,complexity,noise_strength):
    '''
    Generates a Numpy array to specified length and complexity.
    Output array will consist of a combination of sine waves and
    random noise to a specified strength.
    '''

    if noise_strength > 1.0:
        
        raise ValueError("Noise strength should be less than or equal to 1")

    else:

        signal = noise_strength*np.random.random(len(time_array))
        
        if type(complexity) == int:
            coefficients = np.random.random(complexity)
            coefficients = normalise(coefficients)

        else:
            try:
                coefficients = np.random.random(int(complexity))
                coefficients = normalise(coefficients)

            except:
                raise ValueError("Complexity should be an integer")

        freqs = np.random.randint(1,100,size=complexity)

        for c,f in zip(coefficients,freqs):
            
            signal += c*np.sin(f*2*np.pi*time_array)

        print("Frequency values:")
        print(freqs.T)

        filename = ""

        for f in freqs:
            
            filename += str(f) + "_"

        filename = filename[:-1]

        return signal,filename

def normalise(vector):
    '''
    Takes a vector and returns the normalised vector
    '''

    magnitude_s = 0
    for v in vector:
        magnitude_s += v**2

    magnitude = np.sqrt(magnitude_s)

    return vector*(1/magnitude)

def normalise_test():

    vector = np.array([1,2,-1,0],float)

    n = normalise(vector)

    print(n)

def directory_check():

    '''
    Checks if 'Data/' subdirectory exists in current directory.
    If not, 'Data/' subdirectory is created.
    '''

    if not(os.path.isdir("Data")):

        os.system("mkdir Data")

def main():
   
    Npoint = 4096 
    multiple = 500.789

    time = np.linspace(0,10,Npoint*multiple)
    signal,filename = signal_output(time,5,0.5)

    print(filename)

    filename = 'LgUe'

    output_frame = pd.DataFrame({"Time":time, "Signal":signal})

    print("Signal data will be output to 'Data/' directory")

    directory_check()

    output_frame.to_csv("Data/" + filename + ".csv",index=False)

main()

