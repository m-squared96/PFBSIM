#!/usr/bin/python

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal

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

    binsOV = [x for x in range(-N,N,1)]
    binsOV = binsOV[N:] + binsOV[:N]

    binsnOV = [x for x in range(-N//2,N//2,1)]
    binsnOV = binsnOV[N//2:] + binsnOV[:N//2]

    plt.figure()
    plt.plot(binsnOV,toolkit.dB(np.abs(noOverlap_fft.fft/max(noOverlap_fft.fft))),label='No Overlap',c='b',ls='--')
    plt.plot(binsOV,toolkit.dB(np.abs(Overlap_fft.fft/max(Overlap_fft.fft))),label='Overlap',c='r')
    plt.legend()
    plt.title('FFT Bin Response')
    plt.xlabel(r'Bin Offset')
    plt.ylabel('Signal Response (dB)')

def pfb_bin_response(time,window,N,plot_title):
    
    length = 51
    taps = 4

    Sigconfig = {
        'length':length,
        'complexity':1,
        'bandwidth':10,
        'fmax':2000,
        'fmin':0
    }
    
    DSPconfig = {
        'lo':0,
        'mixing':False,
        'lpf':0,
        'ddc':True
    }

    sinc = scipy.signal.firwin(length,taps/length,window='rectangular')
    win = scipy.signal.get_window(window,length)
    signal = sinc*win

    plt.figure()
    for ov_status,col,style in zip((False,True),('b','r'),('--','-')):
        DSPconfig['overlap'] = ov_status
            
        readout = toolkit.FFTGeneric(win,time,N,Sigconfig,DSPconfig)
        response = toolkit.dB(np.abs(readout.fft/max(np.abs(readout.fft))))

        if not readout.overlap:
            bins = [x for x in range(-N//2,N//2,1)]
            #bins = bins[N//2:] + bins[:N//2]
            response = np.concatenate((response[N//2:],response[:N//2]))
            label_str = 'No Overlap'

        else:
            bins = [x for x in range(-N,N,1)]
            #bins = bins[N:] + bins[:N]
            response = np.concatenate((response[N:],response[:N]))
            label_str = 'Overlap'

        plt.plot(bins,response,label=label_str,c=col,ls=style)

    plt.legend()
    plt.title(plot_title)
    plt.xlabel('Bin Offset')
    plt.ylabel('Signal Response (dB)')

def pfb_response_across_windows(time,signal,N,windows,plot_title):
    
    length = 51
    taps = 4

    Sigconfig = {
        'length':len(signal),
        'complexity':1,
        'bandwidth':10,
        'fmax':1,
        'fmin':0,
    }

    DSPconfig = {
        'lo':0,
        'mixing':False,
        'lpf':0,
        'overlap':True,
        'ddc':False
    }
   
    sinc = scipy.signal.firwin(length,taps/length,window='rectangular')
    plt.figure()
    for w,col in zip(windows,('b','r','black')):
        win = scipy.signal.get_window(w,length)
        if w != 'boxcar':
            win *= sinc
            label_str = w.capitalize() + '-Based PFB'
        elif w == 'boxcar':
            label_str = 'Raw FFT'
        readout = toolkit.FFTGeneric(win,time,N,Sigconfig,DSPconfig)
        response = toolkit.dB(np.abs(readout.fft/max(np.abs(readout.fft))))

        if not readout.overlap:
            bins = [x for x in range(-N//2,N//2,1)]
            #bins = bins[N//2:] + bins[:N//2]
            response = np.concatenate((response[N//2:],response[:N//2]))

        else:
            bins = [x for x in range(-N,N,1)]
            #bins = bins[N:] + bins[:N]
            response = np.concatenate((response[N:],response[:N]))

        plt.plot(bins,response,label=label_str,c=col)

    plt.legend()
    plt.title(plot_title)
    plt.xlabel('Bin Offset')
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
        'fmin':0,
        'lut':None
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

    fft_bin_response(time,tophat,Npoint)
    pfb_bin_response(time,'hamming',Npoint,'Hamming-based PFB Bin Response')

    test_windows = ('boxcar','hamming','blackmanharris')
    pfb_response_across_windows(time,tophat,Npoint,test_windows,'Effect of Window Function on Bin Response')

    #fs_attenuation = 1e9
    #time_attenuation = np.arange(0,0.01,1/fs_attenuation)
    #bw_multiples = np.array([100,121.125,142.25,163.375,184.50,205.625,226.75,247.875,269],float)
    #N_attenuation = (1024,2048,4096)
    #
    #for N in N_attenuation:
    #    for ov_status in (True,False):
    #        attenuation_bin_width = fs_attenuation/N
    #        signal_spike_attenuation(time_attenuation,bw_multiples*attenuation_bin_width,N,test_windows,0,ov_status)

main()
plt.show()
