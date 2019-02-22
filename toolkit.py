#!/usr/bin/python3

import numpy as np
from scipy.signal import get_window,firwin

class FilterBank:

    def __init__(self,signal,N,file_length,window):
        '''
        Accepts signal array and returns PFB output of signal.
        
        signal:     Signal data array
        N:          N-point for FFT
        s_length:   Signal branch length
        window:     Window function
        '''

        self.N = N
        self.L = file_length
        self.window = window

        self.signal_array = np.array(signal['Signal'])
        self.time_array = np.array(signal['Time'])

        self.window_coeffs = self.window_function()

    def window_function(self):
        '''
        Accepts the name of a window function and an integer length, and
        returns an array of that length of window coefficients.
        '''

        coefficients = get_window(self.window,self.L)
        return coefficients

    def split(self):
        '''
        Accepts a signal and window function, each of length m*N, and
        divides both into m branches of length N.
        '''

        filtered = np.zeros(self.N)
        m_max = self.L//self.N
        m = 1

        first = 0
        last = self.N

        while m <= m_max:
            
            filtered += self.signal_array[first:last]*self.window_coeffs[first:last]
            
            m += 1
            first = last
            last = m*self.N

        return filtered

    def fft(self,x):
        return np.fft.fft(x,n=self.N)

    def frequencies(self):
        running_interval = 0
        pos = 0

        while pos < len(self.time_array) - 1:
    
            running_interval += self.time_array[pos + 1] - self.time_array[pos]
            pos += 1
    
        sampling_interval = running_interval/(len(self.time_array) - 1)
        sampling_frequency = 1/sampling_interval
        return np.fft.fftfreq(self.N) * sampling_frequency

class FFTGeneric:

    def __init__(self,signal,N,file_length):
        self.N = N
        self.L = file_length

        self.signal_array = np.array(signal['Signal'])
        self.time_array = np.array(signal['Time'])
        

    def fft(self,x):
        return np.fft.fft(x,n=self.N)

    def frequencies(self):
        running_interval = 0
        pos = 0

        while pos < len(self.time_array) - 1:
    
            running_interval += self.time_array[pos + 1] - self.time_array[pos]
            pos += 1
    
        sampling_interval = running_interval/(len(self.time_array) - 1)
        sampling_frequency = 1/sampling_interval
        return np.fft.fftfreq(self.N) * sampling_frequency

def dB(X):
    return 10*np.log10(abs(X))
