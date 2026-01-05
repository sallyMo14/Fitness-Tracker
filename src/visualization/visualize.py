import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df=pd.read_pickle("../../data/interim/01_data_processed.pkl")
df.info()

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------

set_df=df[df['set']==1]
plt.plot(set_df['acc_y'].reset_index(drop=True))


# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

fig=ax=plt.subplots(figsize=(20,6))
label_df=df[df['label']=='dead'].reset_index(drop=True)
plt.plot(label_df[:100]['acc_y'])


labels=df['label'].unique()
for label in labels:
    fig=ax=plt.subplots(figsize=(20,6))
    label_df=df[df['label']==label].reset_index(drop=True)
    plt.plot(label_df[:100]['acc_y'])
    plt.legend([label])



# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
participant= 'A'
label= 'squat'


category_df= df.query(f'label == "{label}"').query(f'participant == "{participant}"').reset_index(drop=True)
fig , ax = plt.subplots(figsize=(20,6))
category_df.groupby(['category'])['acc_y'].plot()
ax.set_ylabel('Acceleration Y-axis')
ax.set_xlabel('Samples')
plt.legend(category_df['category'].unique())



# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

# label , participant , category

label='bench'


participant_df=df.query(f'label =="{label}"').sort_values(by='participant').reset_index(drop=True)
fig , ax = plt.subplots(figsize=(20,6))
participant_df.groupby(['participant'])['acc_y'].plot()
ax.set_ylabel('Acceleration Y-axis')
ax.set_xlabel('Samples')
plt.legend(participant_df['participant'].unique())

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

# plot acc_x , acc_y , acc_z 
label='squat'
participant='A'

subset=df.query(f'label =="{label}" and participant =="{participant}"').reset_index(drop=True)
fig , ax = plt.subplots(figsize=(20,6))
subset[['acc_x','acc_y','acc_z']].plot(ax=ax)
ax.set_ylabel('Acceleration')
plt.legend(['acc_x','acc_y','acc_z'])


# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
labels=df['label'].unique()
participants=df['participant'].unique()

# for accelerometer
for label in labels:
    for participant in participants:
        subset=df.query(f'label =="{label}" and participant =="{participant}"').reset_index(drop=True)
        if subset.empty:
            continue
        fig,ax=plt.subplots(figsize=(20,6))
        subset[['acc_x','acc_y','acc_z']].plot(ax=ax)
        ax.set_title(f'Participant: {participant} - Exercise: {label}')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Acceleration')
        plt.legend(['acc_x','acc_y','acc_z'])
    
    
# for gyroscope    
for label in labels:
    for participant in participants:
        subset=df.query(f'label =="{label}" and participant =="{participant}"').reset_index(drop=True)
        if subset.empty:
            continue
        fig,ax=plt.subplots(figsize=(20,6))
        subset[['gyr_x','gyr_y','gyr_z']].plot(ax=ax)
        ax.set_title(f'Participant: {participant} - Exercise: {label}')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Acceleration')
        plt.legend(['gyr_x','gyr_y','gyr_z'])      
# --------------------------------------------------------------
# Loop over all combinations and   export for both sensors
# --------------------------------------------------------------

for label in labels:
    for participant in participants:
        subset=df.query(f'label =="{label}" and participant =="{participant}"').reset_index(drop=True)
        if subset.empty:
            continue
        fig, axes=plt.subplots(nrows=2,figsize=(20,10), sharex=True)
        subset[['acc_x','acc_y','acc_z']].plot(ax=axes[0])
        subset[['gyr_x','gyr_y','gyr_z']].plot(ax=axes[1])
        axes[0].set_title(f'For acc : Participant: {participant} - Exercise: {label}')
        axes[0].set_ylabel('Acceleration')
        axes[0].legend(['acc_x','acc_y','acc_z'])
        axes[1].set_title(f'For gyr : Participant: {participant} - Exercise: {label}')
        axes[1].set_ylabel('Angular Velocity')
        axes[1].legend(['gyr_x','gyr_y','gyr_z'])
        plt.savefig(f'../../reports/figures/{label.title()}_{participant}_acc_gyr.png')
        
