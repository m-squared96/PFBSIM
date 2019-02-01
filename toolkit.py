#!/usr/bin/python3

import numpy as np
from scipy.signal import get_window,firwin

def dB(X):
    return 10*np.log10(abs(X))

def pfb_init(signal,M,P,window):
    '''
    Accepts signal array and returns PFB output of signal.
    
    signal: Signal data array
    M:      Number of taps
    P:      Number of branches
    W:      Number of windows of length M*P
    window: Window function
    '''

    signal_array = np.array(signal['Signal'])
    time_array = np.array(signal['Time'])

    window_coeffs = window_function(M,P,window)

    W, fir = fir_filter(signal_array,M,P,window_coeffs)

    signal_polyphased = np.fft.fft(fir,P,axis=1)

    return signal_polyphased

def window_function(M,P,window):

    coefficients = get_window(window,M*P)
    sinc = firwin(M*P,cutoff=1.0/P,window='rectangular')

    coefficients *= sinc
    return coefficients

def fir_filter(signal,M,P,window_coeffs):

    if signal.shape[0]%(M*P) != 0:
        
        print("Input signal is not a multiple of taps*branches (" + str(int(M*P)) + ") entries long")
        W = int(signal.shape[0]/M/P - (signal.shape[0]%(M*P))/(M*P))
        print("Concatenating signal length to", str(W*M*P), "values")

    else:
        W = int(signal.shape[0]/M/P)

    signal_p = signal.reshape((W*M,P)).T
    h_p = window_coeffs.reshape((M,P)).T
    summed = np.zeros((P, int(M*W - M)))

    for t in range(0, int(M*W - M)):
        
        weighted = signal_p[:, t:t+M] * h_p
        summed[:,t] = weighted.sum(axis=1)
        
    return W, summed.T
