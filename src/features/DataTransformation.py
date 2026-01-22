from sklearn.decomposition import PCA
import pandas as pd
from scipy.signal import butter, filtfilt, lfilter
import copy

class LowPassFilter:
    

    def low_pass_filter(self,data_table,col,sampling_req,cutoff_freq,order,phase_shift=True):
        
        nyq=0.5*sampling_req # Nyquist Frequency convers freq to Hz (eq : 2000 samples/sec -> nyq=1000 Hz -cycles\sec)
        cut=cutoff_freq/nyq  # Normalized Cutoff Frequency  
        b,a=butter(order,cut,btype='low',analog=False,output='ba') # Butterworth filter design , b : numerator coeff, a: denominator coeff
                    # btype : type of filter(lowpass, highpass, bandpass, bandstop)
                    # analog : False for digital filter
                    # output : 'ba' returns numerator and denominator coeff
                    # order : order of the filter , higher order = sharper cutoff(more aggressive filtering)
        if phase_shift:
            data_table[col+'_lp']=filtfilt(b,a,data_table[col])
        else:
            data_table[col+'_lp']=lfilter(b,a,data_table[col])
        return data_table
        