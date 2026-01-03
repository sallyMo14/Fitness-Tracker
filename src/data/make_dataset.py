import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------


single_acc_file=pd.read_csv("../../data/raw/MetaMotion/MetaMotion\\A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")
single_acc_file.head()


# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/MetaMotion/*.csv")
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
f=files[0]
data_path="../../data/raw/MetaMotion/MetaMotion\\"
filename=f.replace(data_path,"")
filename_parts=filename.split("-")

participant=filename_parts[0]
label=filename_parts[1]  
category=filename_parts[2].rstrip('123')

df=pd.read_csv(f)
df['participant']=participant
df['label']=label
df['category']=category

# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df=pd.DataFrame()
gyr_df=pd.DataFrame()
acc_set=1
gyr_set=1

for f in files :
    filename=f.replace(data_path,"")
    filename_parts=filename.split("-")

    participant=filename_parts[0]
    label=filename_parts[1]  
    category=filename_parts[2].rstrip('123')

    df=pd.read_csv(f)
    df['participant']=participant
    df['label']=label
    df['category']=category
    if "Accelerometer" in f :
        df['set']=acc_set
        acc_df=pd.concat([acc_df,df])
        acc_set+=1
    if "Gyroscope" in f :
        df['set']=gyr_set
        gyr_df=pd.concat([gyr_df,df])
        gyr_set+=1
 
acc_df.shape
acc_df.head()
gyr_df.head()

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
acc_df.info()
pd.to_datetime(df['epoch (ms)'], unit='ms')

acc_df.index
acc_df.index=pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
gyr_df.index=pd.to_datetime(gyr_df['epoch (ms)'], unit='ms')

acc_df.index
acc_df.head()

del acc_df['epoch (ms)']
del acc_df['elapsed (s)']
del acc_df['time (01:00)']
acc_df.head()

del gyr_df['epoch (ms)']
del gyr_df['elapsed (s)']
del gyr_df['time (01:00)']
gyr_df.head()

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/MetaMotion/*.csv")
data_path="../../data/raw/MetaMotion/MetaMotion\\"

def read_data_from_files(files):
    
    acc_df=pd.DataFrame()
    gyr_df=pd.DataFrame()
    
    acc_set=1
    gyr_set=1

    for f in files :
        filename=f.replace(data_path,"")
        filename_parts=filename.split("-")

        participant=filename_parts[0]
        label=filename_parts[1]  
        category=filename_parts[2].rstrip('123')

        df=pd.read_csv(f)
        
        df['participant']=participant
        df['label']=label
        df['category']=category
        
        if "Accelerometer" in f :
            df['set']=acc_set
            acc_df=pd.concat([acc_df,df])
            acc_set+=1
        if "Gyroscope" in f :
            df['set']=gyr_set
            gyr_df=pd.concat([gyr_df,df])
            gyr_set+=1
            
    acc_df.index=pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
    gyr_df.index=pd.to_datetime(gyr_df['epoch (ms)'], unit='ms')

    del acc_df['epoch (ms)']
    del acc_df['elapsed (s)']
    del acc_df['time (01:00)']

    del gyr_df['epoch (ms)']
    del gyr_df['elapsed (s)']
    del gyr_df['time (01:00)']
    
    return acc_df, gyr_df

acc_df, gyr_df = read_data_from_files(files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
data_merged = pd.concat([acc_df.iloc[:,:3],gyr_df],axis=1)
data_merged.head()

data_merged.columns=[
    'acc_x',
    'acc_y',
    'acc_z',
    'gyr_x',
    'gyr_y',
    'gyr_z',
    'participant',
    'label',
    'category',
    'set'
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz 

resampling={
    "acc_x":"mean",
    "acc_y":"mean",
    "acc_z":"mean",
    "gyr_x":"mean",
    "gyr_y":"mean",
    "gyr_z":"mean",
    "participant":"last",
    "label":"last",
    "category":"last",
    "set":"last"
}
#split by days to avoid resampling over day limits
days = [x for n,x in data_merged.groupby(pd.Grouper(freq='D'))]
data_resampled=pd.concat([df.resample(rule='200ms').apply(resampling).dropna() for df in days])

data_resampled.info()
data_resampled['set']=data_resampled['set'].astype(int)

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")