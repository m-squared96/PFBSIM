#!/usr/bin/python3

import numpy as np
from scipy.signal import get_window,firwin,butter,sosfilt

#TODO: Split FFTs

class FilterBank:

    def __init__(self,signal,N,file_length,window,lo,lut):
        '''
        Accepts signal array and returns PFB output of signal.
        
        signal:         Signal data array
        N:              N-point for FFT
        file_length:    Length of signal/time arrays
        window:         Window function
        lo:             IQ mixer local oscillator frequency
        '''

        self.N = N
        self.L = file_length
        self.window = window
        self.lo = lo
        self.lut = lut

        self.signal_array = np.array(signal['Signal'])
        self.time_array = np.array(signal['Time'])

        self.window_coeffs = self.window_function()
        self.fs = fs_finder(self.time_array)

        self.iq()
        self.I_fir,self.Q_fir = self.split(self.I),self.split(self.Q)
        self.I_fft,self.Q_fft = fft(self.I_fir,N=self.N),fft(self.Q_fir,N=self.N)

        self.temp_split = self.split(self.signal_array)
        self.temp_fft = fft(self.temp_split,N=self.N)

    def window_function(self):
        '''
        Accepts the name of a window function and an integer length, and
        returns an array of that length of window coefficients.
        '''

        coefficients = get_window(self.window,self.L)
        sinc = firwin(self.L,cutoff=1.0/self.N,window="rectangular")
        coefficients *= sinc
        return coefficients

    def split(self,x):
        '''
        Accepts a signal and window function, each of length m*N, and
        divides both into m branches of length N.
        '''

        x = np.array(x)

        filtered = np.zeros(self.N)
        m_max = self.L//self.N
        m = 1

        first = 0
        last = self.N

        while m <= m_max:
            
            filtered += x[first:last]*self.window_coeffs[first:last]
            
            m += 1
            first = last
            last = m*self.N

        return filtered

    def iq(self):
        self.I,self.Q = iq_mixer(self.signal_array,2.2e9,self.time_array,self.fs)

class FFTGeneric:

    def __init__(self,signal,N,file_length,lo):
        self.N = N
        self.L = file_length
        self.lo = lo

        self.signal_array = np.array(signal['Signal'])
        self.time_array = np.array(signal['Time'])
        self.fs = fs_finder(self.time_array)

        self.iq()
        self.I_fft,self.Q_fft = fft(self.I,N=self.N),fft(self.Q,N=self.N)

    def iq(self):
        self.I,self.Q = iq_mixer(self.signal_array,2.2e9,self.time_array,self.fs)

def dB(X):
    return 10*np.log10(abs(X))

def iq_mixer(signal,lo,time_array,fs):
    inphase = signal*np.cos(2*np.pi*lo*time_array)
    quadrature = -1*signal*np.sin(2*np.pi*lo*time_array)
    return inphase,quadrature

def lpf(signal,cutoff,fs):
    lowpass = butter(10,cutoff,btype='low',analog=False,output='sos',fs=fs)
    return sosfilt(lowpass,signal)

def fs_finder(time_array):
    running_interval = 0
    pos = 0

    while pos < len(time_array) - 1:

        running_interval += time_array[pos + 1] - time_array[pos]
        pos += 1

    sampling_interval = running_interval/(len(time_array) - 1)
    sampling_frequency = 1/sampling_interval
    return sampling_frequency

def fft(x,N):
    return np.fft.fft(x,n=N)

def channel_selector(fft_signal,freqs_array,no_channels):
    '''
    Accepts FFT output (from -fmax < 0 < fmax) and evenly divides
    this signal into an evenly spaced, specified number of channels

    fft_signal:         Numpy FFT output
    freqs_array:        Numpy fftfreq output
    no_channels:        Desired number of evenly-spaced channels
    '''
    fft_signal = np.array(fft_signal)

    print("x")
