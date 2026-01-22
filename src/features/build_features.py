import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter#, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df=pd.read_pickle("../../data/interim/02_data_outliers_removed.pkl")
df.head()

#styling
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 6)
plt.rcParams['figure.dpi'] = 100

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
df.info()
df.isna().sum()
predictor_cols=list(df.columns[:6])

for col in predictor_cols:
    df[col].interpolate(method='linear', inplace=True)


# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------

duration = df[df['set']==1].index[-1] - df[df['set']==1].index[0]
duration.seconds

for s in df['set'].unique():
    start=df[df['set']==s].index[0]
    end=df[df['set']==s].index[-1]
    duration=end - start
    df.loc[df['set']==s, 'duration']=duration.seconds

#since each excersise have multiple categories we want to calculate the average duration per category
duration_df=df.groupby('category')['duration'].mean()

#calculate the duration for each repetition
duration_df.iloc[0]/5 # heavy sets have 5 repetitions
duration_df.iloc[1]/10 # medium sets have 10 repetitions
# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

# REMOVES HIGH FREQUENCY NOISE FROM THE SIGNAL
# Its a type of signal processing designed to have a greq response as flat as possible in the pass band
# keep low frequency components and flatten high frequency components
df_lp=df.copy()
lpf=LowPassFilter()
sf=1000/200  # sampling frequency = 5 Hz ( 1000 ms / 200 ms )
cutoff=1.3# cutoff frequency = 1 Hz (cutoff should be less than half of sampling frequency (Nyquist Theorem) )
# -----------> tuning parameter until we find good balance a smooth signal and a characteristic pattern for each exercise (via visualization)
# the higher the cutoff frequency the less smooth the signal will be
# the choosen cutoff will effect prediction  results
order=5    # order of the filter
for col in predictor_cols:
    df_lp=lpf.low_pass_filter(data_table=df_lp,col=col,sampling_req=sf,cutoff_freq=cutoff,order=order,phase_shift=True)
 
subset=df_lp[df_lp['set']==30]
fig,axes=plt.subplots(2,1,figsize=(20,10),sharey=True,sharex=True)   
axes[0].plot(subset['acc_y'], label='Original Signal')
axes[1].plot(subset['acc_y_lp'], label='Low-pass Filtered Signal')
axes[0].set_title(f'for excercise : {subset["label"].iloc[0]} \n\nOriginal Signal - acc_y ')
axes[1].set_title('Low-pass Filtered Signal - acc_y')

df_lp=df.copy()
for col in predictor_cols:
    # apply low pass filter and override the columns 
    df_lp=lpf.low_pass_filter(data_table=df_lp,col=col,sampling_req=sf,cutoff_freq=cutoff,order=order,phase_shift=True)
    df_lp[col] = df_lp[col+'_lp']
    df_lp = df_lp.drop(col+'_lp', axis=1)
    
    
 
# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------