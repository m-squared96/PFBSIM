#!/usr/bin/python3

import numpy as np
from scipy.signal import get_window,firwin,iirfilter,sosfilt

class FilterBank:

    def __init__(self,signal,time,PFBdict,Sigdict,DSPdict):
        '''
        Accepts signal array and returns PFB output of signal.
        
        signal:         Signal data array
        N:              N-point for FFT
        file_length:    Length of signal/time arrays
        window:         Window function
        lo:             IQ mixer local oscillator frequency
        '''
        # PFBdict parameters
        self.N = PFBdict['N']
        self.taps = PFBdict['taps']
        self.window = PFBdict['window']
        # Sigdict parameters
        self.L = Sigdict['length']
        self.lut = Sigdict['lut']
        self.complexity = Sigdict['complexity']
        self.bandwidth = Sigdict['bandwidth']
        self.fmax = Sigdict['fmax']
        self.fmin = Sigdict['fmin']
        # DSPdict parameters
        self.lo = DSPdict['lo']
        self.mixing = DSPdict['mixing']
        self.lpf_cutoff = DSPdict['lpf']
        self.overlap = DSPdict['overlap']

        # Data and variables derived from the above

        if type(signal) is np.ndarray:
            self.signal_array = signal
        elif type(signal) is not np.ndarray:
            self.signal_array = np.array(signal)

        if type(time) is np.ndarray:
            self.time_array = time
        elif type(time) is not np.ndarray:
            self.time_array = np.array(time)

        self.window_coeffs = self.window_function()
        self.fs = fs_finder(self.time_array)
        self.iq()

        print("\nBeginning coarse channelisation stage")
            
        self.I_fir,self.Q_fir = self.split(self.I),self.split(self.Q)
        self.I_fft,self.Q_fft = fft(self.I_fir,N=self.N),fft(self.Q_fir,N=self.N)

        if self.overlap:
            self.bin_width = self.fs/(self.N*2)
            self.fft = self.fft_overlap()
            self.freqs = np.fft.fftfreq(n=self.N*2)*self.fs

        else:
            self.bin_width = self.fs/self.N
            self.fft = self.I_fft
            self.freqs = np.fft.fftfreq(n=self.N)*self.fs
        
        print("Coarse channelisation complete")

        self.fine_channelisation(self.complexity)
        
        #if self.lut is not None:
        #    print('Phase conversion stage')
        #    self.phase_conversion()

    def window_function(self):
        '''
        Accepts the name of a window function and an integer length, and
        returns an array of that length of window coefficients.
        '''

        coefficients = get_window(self.window,self.L)
        sinc = firwin(self.L,cutoff=self.taps/self.L,window="rectangular")
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
        if self.mixing:
            self.I,self.Q = iq_mixer(self.signal_array,self.lo,self.time_array,self.fs,self.lpf_cutoff)
        else:
            self.I,self.Q = self.signal_array,self.signal_array

    def fft_overlap(self):
        fft = []
        for i in range(self.N):
            fft.append(self.I_fft[i])
            fft.append(self.Q_fft[i])

        return np.array(fft)

    def fine_channelisation(self,no_channels):
        if self.lut is None:
            print('\nFine channelisation cannot occur due to issues with LUT.')

        elif self.lut is not None:
            print('\nBeginning fine channelisation')
            
            if self.mixing:
                start_freq = self.fmin - self.lo
                start_bin = int(start_freq/self.bin_width)

            elif not self.mixing:
                start_freq = self.fmin
                start_bin = int(start_freq/self.bin_width) - 1

            end_freq = start_freq + self.bandwidth            
            end_bin = int(end_freq/self.bin_width) + 3

            pos_freqs = self.freqs[start_bin:end_bin]
            pos_fft = self.fft[start_bin:end_bin]
            
            # Bins = FFT signal divided into bins
            # Bin_frequencies = Frequency array divided into bins
            bins,bin_frequencies = channel_selector(pos_fft,pos_freqs,no_channels)
            fpeak_list = []
            for i in range(bins.shape[0]):
                bin_fpeak_pos = int(bins[i].argmax())
                fpeak_list.append(bin_frequencies[i][bin_fpeak_pos])

            singlespike_signals = self.frequency_isolation(fpeak_list)
            ddc_bins = self.digital_down_conversion(singlespike_signals)
            zero_hz_amplitudes = []
            for row in range(ddc_bins.shape[0]):
                zero_hz_amplitudes.append(np.abs(np.fft.fft(ddc_bins[row],n=self.N)[0]))

            print('Fine channelisation stage complete')

    def frequency_isolation(self,fpeaks):
        bandpass_list = []
        for f in fpeaks:
            bandpass_list.append((f - 0.25e6, f + 0.25e6))

        singlespike_signals = np.empty((0,len(self.signal_array)),float)
        for bp_range in bandpass_list:
            singlespike_signals = np.vstack([singlespike_signals,bpf(self.signal_array,bp_range,self.fs)])

        return singlespike_signals

    def digital_down_conversion(self, singlespikes):
        double_mixed = singlespikes*singlespikes
        ddc_bins = np.empty((0, len(double_mixed[0])),float)

        for sig in range(double_mixed.shape[0]):
            ddc_bins = np.vstack([ddc_bins,lpf(double_mixed[sig],0.25e6,self.fs)])
        return ddc_bins

    def phase_conversion(self):
       bin_phase_shifts = []
       for I,Q in zip(self.IQ_zero_hz_amplitudes['I'], self.IQ_zero_hz_amplitudes['Q']):
           bin_phase_shifts.append(np.arctan(Q/I))

       self.bin_phase_shifts = bin_phase_shifts


class FFTGeneric:

    def __init__(self,signal,time,N,Sigdict,DSPdict):
        
        self.N = N

        # Sigdict parameters
        self.L = Sigdict['length']
        # DSPdict parameters
        self.lo = DSPdict['lo']
        self.mixing = DSPdict['mixing']
        self.lpf_cutoff = DSPdict['lpf']
        self.overlap = DSPdict['overlap']

        # Data derived from above parameters
        if type(signal) is np.ndarray:
            self.signal_array = signal
        elif type(signal) is not np.ndarray:
            self.signal_array = np.array(signal)

        if type(time) is np.ndarray:
            self.time_array = time
        elif type(time) is not np.ndarray:
            self.time_array = np.array(time)

        self.fs = fs_finder(self.time_array)
        self.iq()
        self.I_fft,self.Q_fft = fft(self.I,N=self.N),fft(self.Q,N=self.N)

        if self.overlap:
            self.bin_width = self.fs/(self.N*2)
            self.fft = self.fft_overlap()
            self.freqs = np.fft.fftfreq(n=self.N*2)*self.fs

        else:
            self.bin_width = self.fs/self.N
            self.fft = self.I_fft
            self.freqs = np.fft.fftfreq(n=self.N)*self.fs

    def iq(self):
        if self.mixing:
            self.I,self.Q = iq_mixer(self.signal_array,self.lo,self.time_array,self.fs,self.lpf_cutoff)

        else:
            self.I,self.Q = self.signal_array,self.signal_array

    def fft_overlap(self):
        fft = []
        for i in range(self.N):
            fft.append(self.I_fft[i])
            fft.append(self.Q_fft[i])

        return np.array(fft)

def dB(X):
    return 10*np.log10(abs(X))

def iq_mixer(signal,lo,time_array,fs,lpf_cutoff):
    inphase = lpf(signal*np.cos(2*np.pi*lo*time_array),lpf_cutoff,fs)
    quadrature = lpf(-1*signal*np.sin(2*np.pi*lo*time_array),lpf_cutoff,fs)
    return inphase,quadrature

def lpf(signal,cutoff,fs):
    lowpass = iirfilter(17,cutoff,btype='low',output='sos',fs=fs)
    return sosfilt(lowpass,signal)

def hpf(signal,cutoff,fs):
    highpass = iirfilter(17,cutoff,btype='high',output='sos',fs=fs)
    return sosfilt(highpass,signal)

def bpf(signal,f_range,fs):
    bandpass = iirfilter(17,f_range,btype='band',output='sos',fs=fs)
    return sosfilt(bandpass,signal)

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
        print("Channel selection could not occur. Lengths of FFT and frequency arrays do not match")
        return 0,0
