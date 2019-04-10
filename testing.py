#!/usr/bin/python

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import chirp

import toolkit

np.seterr(divide = 'ignore')

def fft_bin_response(time,signal,N):

    Sigconfig = {
        'length':len(time)
    }

    DSPconfig_noOverlap = {
        'lo':0,
        'mixing':False,
        'lpf':0,
        'overlap':False
    }
    
    DSPconfig_Overlap = {
        'lo':0,
        'mixing':False,
        'lpf':0,
        'overlap':True
    }

    noOverlap_fft = toolkit.FFTGeneric(signal,time,N,Sigconfig,DSPconfig_noOverlap)
    Overlap_fft = toolkit.FFTGeneric(signal,time,N,Sigconfig,DSPconfig_Overlap)

    plt.figure()
    plt.plot(noOverlap_fft.freqs,toolkit.dB(np.abs(noOverlap_fft.fft/max(noOverlap_fft.fft))),label='No Overlap')
    plt.title('FFT Bin Response')
    plt.xlabel(r'Frequency (Hz)')
    plt.ylabel('Signal Response (dB)')

def pfb_bin_response(time,signal,N):

    PFBconfig = {
        'N':N,
        'taps':4,
        'window':'hamming'
    }

    Sigconfig = {
        'length':4096,
        'complexity':1,
        'bandwidth':10,
        'fmax':1,
        'fmin':0
    }

    DSPconfig_noOverlap = {
        'lo':0,
        'mixing':False,
        'lpf':0,
        'overlap':False,
        'ddc':False
    }
    
    DSPconfig_Overlap = {
        'lo':0,
        'mixing':False,
        'lpf':0,
        'overlap':True,
        'ddc':False
    }
    
    noOverlap_PFB = toolkit.FilterBank(signal,time,PFBconfig,Sigconfig,DSPconfig_noOverlap)
    Overlap_PFB = toolkit.FilterBank(signal,time,PFBconfig,Sigconfig,DSPconfig_Overlap)

    plt.figure()
    plt.plot(noOverlap_PFB.freqs,toolkit.dB(np.abs(noOverlap_PFB.fft)/(len(signal)/2)),label='No Overlap')
    plt.title('PFB Bin Response')
    plt.xlabel(r'Frequency (Hz)')
    plt.ylabel('Signal Response (dB)')

def pfb_response_across_windows(time,signal,N,windows,plot_title):

    PFBconfig = {
        'N':N,
        'taps':4
    }

    Sigconfig = {
        'length':len(signal),
        'complexity':1,
        'bandwidth':10,
        'fmax':1,
        'fmin':0
    }

    DSPconfig = {
        'lo':0,
        'mixing':False,
        'lpf':0,
        'overlap':False,
        'ddc':False
    }

    plt.figure()

    for w in windows:
        PFBconfig['window'] = w
        pfbmodel = toolkit.FilterBank(signal,time,PFBconfig,Sigconfig,DSPconfig)
        plt.plot(pfbmodel.freqs,toolkit.dB(np.abs(pfbmodel.fft)/(len(signal)/2)),label=w.capitalize())

    plt.title(plot_title)
    plt.legend()
    plt.xlabel(r'Frequency (Hz)')
    plt.ylabel('Signal Response (dB)')

def signal_spike_attenuation(time,signal_frequencies,N,windows,noise,overlap=False):

    if noise != 0:
        signal = np.random.random(len(time))*noise

    elif noise == 0:
        signal = np.zeros(len(time))
    
    for f in signal_frequencies:
        signal += np.sin(2*np.pi*f*time)
    
    PFBconfig = {
        'N':N,
        'taps':4
    }

    Sigconfig = {
        'length':len(signal),
        'complexity':1,
        'bandwidth':10,
        'fmax':1,
        'fmin':0
    }

    DSPconfig = {
        'lo':0,
        'mixing':False,
        'lpf':0,
        'overlap':overlap,
        'ddc':False
    }

    raw_fft = toolkit.FFTGeneric(signal,time,N,Sigconfig,DSPconfig)
    plt.figure()
    plt.plot(raw_fft.freqs,np.abs(raw_fft.fft),label='Raw FFT')

    for w in windows:
        PFBconfig['window'] = w
        pfb = toolkit.FilterBank(signal,time,PFBconfig,Sigconfig,DSPconfig)
        plt.plot(pfb.freqs,np.abs(pfb.fft)*1e6,label='PFB ' + w.capitalize())

    plt.title('Bin offset attenuation, N-point: ' + str(N) + 'Overlap: ' + str(overlap))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Signal Strength')
    plt.legend()

def pulse(width,total_length):
    
    sig_pulse = np.ones(width)

    if width == total_length:
        return sig_pulse

    else:
        if width%2 == 0:
            z = np.zeros((total_length - width)//2)
            return np.concatenate((sig_pulse,z,z))

        else:
            z1 = np.zeros((total_length - width)//2)
            z2 = np.zeros((total_length - width)//2 + 1)
            return np.concatenate((sig_pulse,z1,z2))

def main():
    
    Npoint = 4096
    time = np.linspace(0,0.1,4096)
    
    tophat = pulse(10,Npoint)
    delta = pulse(1,Npoint)
    #dc = pulse(Npoint*100,Npoint*100)
    #chirp_sig = chirp(time,0,time[-1],100,method='linear')

    fft_bin_response(time,tophat,Npoint)
    pfb_bin_response(time,tophat,Npoint)

    test_windows = ('hamming','hann','blackman','boxcar')
    #pfb_response_across_windows(time,tophat,Npoint,test_windows,'Tophat Frequency Response')
    #pfb_response_across_windows(time,dc,Npoint,test_windows,'DC Frequency Response')
    #pfb_response_across_windows(time,chirp_sig,Npoint,test_windows,'Chirp Frequency Response')

    #fs_attenuation = 1e9
    #time_attenuation = np.arange(0,0.01,1/fs_attenuation)
    #bw_multiples = np.array([100,121.125,142.25,163.375,184.50,205.625,226.75,247.875,269],float)
    #N_attenuation = (1024,2048,4096)
    
#    for N in N_attenuation:
#        for ov_status in (True,False):
#            attenuation_bin_width = fs_attenuation/N
#            signal_spike_attenuation(time_attenuation,bw_multiples*attenuation_bin_width,N,test_windows,0,ov_status)
#
main()
plt.show()
