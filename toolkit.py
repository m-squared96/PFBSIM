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

        self.filtered = filtered

    def fft(self):
        self.signal_f = np.fft.fft(self.filtered,n=self.N)

    def frequencies(self):
        t_space = self.time_array[1] - self.time_array[0]
        self.freqs = np.linspace(0.0,1/2*t_space,self.N//2)

def dB(X):
    return 10*np.log10(abs(X))
