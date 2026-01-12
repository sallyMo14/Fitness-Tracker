import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df= pd.read_pickle("../../data/interim/01_data_processed.pkl")
df.info()

# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 6)
plt.rcParams['figure.dpi'] = 100
df[['acc_y']].boxplot()
df[['acc_x','label']].boxplot(by='label')
outliers_cols=df.columns[:6]

    # for acc data
df[[*outliers_cols[:3] ,'label']].boxplot(by='label',layout=(3,3),figsize=(12,12))

    # for gyr data
df[[*outliers_cols[3:] ,'label']].boxplot(by='label',layout=(3,3),figsize=(12,12))


# Function to plot outliers in a binary fashion
def plot_binary_outliers(dataset , col , outlier_col , reset_index=True,title=f"Outliers in {col}"):
    """
    Plots the data points in a column, coloring the outliers differently.
    
    Parameters:
    dataset (pd.DataFrame): The input dataframe containing the data.
    col (str): The name of the column to plot.
    outlier_col (str): The name of the column indicating outliers (True for outlier, False for normal).
    reset_index (bool): Whether to reset the index for plotting.
    """
    if reset_index:
        dataset=dataset[[col, outlier_col]].reset_index()
    
    fig,ax= plt.subplots()
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.title(title)
    # plot non outliers data point in default color
    ax.plot(dataset.index[~dataset[outlier_col]], 
            dataset[col][~dataset[outlier_col]],
            '+')
    
    # plot outliers data point in red color
    ax.plot(dataset.index[dataset[outlier_col]], 
            dataset[col][dataset[outlier_col]],
            'r+')
    
    plt.legend(['outliers'+col , 'non-outliers'+col])
    plt.show()

# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------


# Insert IQR function

def mark_outliers_iqr(dataset, col):
    """
    Marks outliers in a specified column of the dataset using the Interquartile Range (IQR) method.
    
    Parameters:
    dataset (pd.DataFrame): The input dataframe containing the data.
    col (str): The name of the column to check for outliers.
    
    Returns:
    The dataframe with an additional boolean column indicating outliers.
    """
    dataset = dataset.copy()
    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    dataset[col+'_outlier'] = (dataset[col] < lower_bound) | (dataset[col] > upper_bound)
    return dataset

# Plot a single column
col='acc_x'
dataset = mark_outliers_iqr(df, col)
plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+'_outlier', reset_index=True)

# Loop over all columns
for col in outliers_cols:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+'_outlier', reset_index=True)
    

# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

# Check for normal distribution
df[[*outliers_cols[:3] ,'label']].hist(by='label', layout=(3,3), figsize=(12,12))


# Insert Chauvenet's function

def mark_outliers_chauvenet(dataset, col ,c=2):
    """
    Marks outliers in a specified column of the dataset using Chauvenet's criterion.
    
    Parameters:
    dataset (pd.DataFrame): The input dataframe containing the data.
    col (str): The name of the column to check for outliers.
    c (float): The Chauvenet's criterion constant.
    
    Returns:
    The dataframe with an additional boolean column indicating outliers.
    """
    dataset=dataset.copy()
    mean= dataset[col].mean()
    std=dataset[col].std()
    N= len(dataset)
    deviation = (dataset[col]-mean) / std 
    criterion = 1.0 / (c*N)
    
    low=-deviation / math.sqrt(2)
    high=deviation / math.sqrt(2)
    
    prob=[]
    mask=[]
    
    for i in range(N):
        prob.append( 1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i])) )
        mask.append(prob[i] < criterion)
        
    dataset[col+'_outlier'] = mask
    return dataset

# Loop over all columns

for col in outliers_cols:
    dataset=mark_outliers_chauvenet(df, col)
    plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+'_outlier', reset_index=True)

# --------------------------------------------------------------
# Local outlier factor (distance based)(unsupervised technique)
# --------------------------------------------------------------

# Insert LOF function

def mark_outliers_lof(dataset, col, n_neighbors=20):
    """
    Marks outliers in a specified column of the dataset using the Local Outlier Factor (LOF) method.
    
    Parameters:
    dataset (pd.DataFrame): The input dataframe containing the data.
    col (str): The name of the column to check for outliers.
    n_neighbors (int): The number of neighbors to use for LOF.
    
    Returns:
    The dataframe with an additional boolean column indicating outliers.
    """
    dataset=dataset.copy()
    lof=LocalOutlierFactor(n_neighbors=n_neighbors)
    data=dataset[[col]]
    outliers =lof.fit_predict(data)
    X_scores=lof.negative_outlier_factor_
    
    dataset[col+'_outlier'] = outliers == -1
    return dataset
# Loop over all columns
for col in outliers_cols:
    dataset= mark_outliers_lof(df, col)
    plot_binary_outliers(dataset,col,col+'_outlier', reset_index=True)
    
    # Note : Adjust n_neighbors parameter if needed for better results
    # here the outliers are within the range of other normal data points , not isolated like previous methods
# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------
for label in df['label'].unique():
    for col in outliers_cols:
        subset=df[df['label']==label]
        dataset=mark_outliers_lof(subset, col)
        plot_binary_outliers(dataset,col,col+'_outlier', reset_index=True , title ="Outliers in "+ col + " for label: "+ label)
        

# --------------------------------------------------------------
# Choose method and deal with outliers --> Chauvenets because it give a reasanable no of outliers 
# --------------------------------------------------------------

# Test on single column
col="acc_x"
dataset=mark_outliers_chauvenet(df,col)
subset = dataset[dataset[col+'_outlier']]
dataset.loc[subset.index,col]=np.nan
dataset[col].isna().sum()
len(subset)

    
# Create a loop
outlier_removed_df=df.copy()
for col in outliers_cols:
    for label in df['label'].unique():
        dataset = mark_outliers_chauvenet(df[df['label'] == label], col)
        mask = dataset[col + '_outlier']  
        dataset.loc[mask, col] = np.nan
        outlier_removed_df.loc[outlier_removed_df['label'] == label, col] = dataset[col]
        n_outliers = int(mask.sum())
        print(f'For column: {col} and label: {label} , number of outliers removed: {n_outliers}')

outlier_removed_df.info()

# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------
outlier_removed_df.to_pickle("../../data/interim/02_data_outliers_removed.pkl")