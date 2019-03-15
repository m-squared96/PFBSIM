#!/usr/bin/python3

import numpy as np
from scipy.signal import get_window,firwin,iirfilter,sosfilt

#TODO: Split FFTs

class FilterBank:

    def __init__(self,signal,N,file_length,window,lo,lut,lpf_cutoff,taps):
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
        self.lpf_cutoff = lpf_cutoff
        self.lut = lut
        self.taps = taps

        self.signal_array = np.array(signal['Signal'])
        self.time_array = np.array(signal['Time'])

        self.window_coeffs = self.window_function()
        self.fs = fs_finder(self.time_array)

        self.iq()

        print("\nBeginning coarse channelisation stage")
            
        self.I_fir,self.Q_fir = self.split(self.I),self.split(self.Q)
        self.I_fft,self.Q_fft = fft(self.I_fir,N=self.N),fft(self.Q_fir,N=self.N)
        self.freqs = np.fft.fftfreq(n=self.N)*self.fs
        
        print("Coarse channelisation complete")

        self.fine_channelisation(self.N//2)

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
        x = np.reshape(x,(self.taps,self.L//self.taps))
        w = np.reshape(self.window_coeffs,(self.taps,self.L//self.taps))
        x_win = x*w
        return np.sum(x_win,axis=0)

    def iq(self):
        self.I,self.Q = iq_mixer(self.signal_array,self.lo,self.time_array,self.fs,self.lpf_cutoff)

    def fine_channelisation(self,no_channels):
        if self.lut is None:
            print('\nFine channelisation cannot occur due to issues with LUT.')

        elif self.lut is not None:
            print('\nBeginning fine channelisation')
            bins,bin_frequencies = channel_selector((self.I_fft + self.Q_fft),self.freqs,no_channels)

class FFTGeneric:

    def __init__(self,signal,N,file_length,lo,lpf_cutoff):
        self.N = N
        self.L = file_length
        self.lo = lo
        self.lpf_cutoff = lpf_cutoff

        self.signal_array = np.array(signal['Signal'])
        self.time_array = np.array(signal['Time'])
        self.fs = fs_finder(self.time_array)

        self.iq()
        self.I_fft,self.Q_fft = fft(self.I,N=self.N),fft(self.Q,N=self.N)

    def iq(self):
        self.I,self.Q = iq_mixer(self.signal_array,self.lo,self.time_array,self.fs,self.lpf_cutoff)

def dB(X):
    return 10*np.log10(abs(X))

def iq_mixer(signal,lo,time_array,fs,lpf_cutoff):
    inphase = lpf(signal*np.cos(2*np.pi*lo*time_array),lpf_cutoff,fs)
    quadrature = lpf(-1*signal*np.sin(2*np.pi*lo*time_array),lpf_cutoff,fs)
    return inphase,quadrature

def lpf(signal,cutoff,fs):
    lowpass = iirfilter(17,cutoff,btype='low',output='sos',fs=fs)
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
    freqs_array = np.array(fft_signal)

    l_fft = np.shape(fft_signal)[0]
    l_freqs = np.shape(freqs_array)[0]

    if l_fft == l_freqs:
        if l_fft % no_channels == 0:
            bins = np.reshape(fft_signal,(no_channels,l_fft//no_channels))
            freq_bins = np.reshape(freqs_array,(no_channels,l_freqs//no_channels))

        elif l_fft % no_channels != 0:
            elements_needed = no_channels - (l_fft % no_channels)
            
            pad_forfft = np.zeros(elements_needed)
            padded_fft = np.concatenate((fft_signal,pad_forfft))
            bins = np.reshape(padded_fft,(no_channels,(padded_fft.shape[0])//no_channels))

            freq_spacing = abs(freqs_array[1] - freqs_array[0])
            pad_forfreq = []
            count = 1
            while count <= elements_needed:
                pad_forfreq.append(max(freqs_array) + count*freq_spacing)
                count += 1

            padded_freqs = np.concatenate((freqs_array,pad_forfreq))
            freq_bins = np.reshape(padded_freqs,(no_channels,(padded_freqs.shape[0])//no_channels))

        return bins,freq_bins

    elif l_fft != l_freqs:
        print("Fine channelisation could not occur. Lengths of FFT and frequency arrays do not match")
        return 0,0
