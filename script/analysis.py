import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
sns.set_theme(style="whitegrid")
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})


def average_load_profile():
    
    
    
    
    
    return None

def diversity_factor():
    
    
    
    return None

def clustering():
    
    
    
    return None

def seasonality_decomposition(input_df):
    
    temp_df = input_df
    temp_df.set_index('Datetime', inplace=True)
    for column in temp_df.columns:
        plot_df = temp_df[[column]]
        decompose_result_mult = seasonal_decompose(plot_df, model="multiplicative")
        
        trend = decompose_result_mult.trend
        seasonal = decompose_result_mult.seasonal
        residual = decompose_result_mult.resid

        decompose_result_mult.plot()
        plt.show()
    
    
    return None

def socio_economic_analysis():
    
    
    return None


def weather_analysis():
    
    
    
    return None
    