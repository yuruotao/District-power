import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os

sns.set_theme(style="whitegrid")
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})


def average_load_profile():
    
    
    
    
    
    return None

def diversity_factor(input_df, output_path):
    
    datetime_column = input_df["Datetime"]
    temp_df = input_df.set_index(["Datetime"])
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Calculate total load for each time step
    total_load = temp_df.drop(columns=[datetime_column]).sum(axis=1)
    
    # Calculate maximum load
    max_load = total_load.max()
    
    # Calculate diversity factor
    diversity_factor = total_load / max_load
    
    print(diversity_factor)
    
    return None

def clustering():
    
    
    
    return None

def seasonality_decomposition(input_df, output_path, period_num, model):
    """decomposition the dataframe by seasonality

    Args:
        input_df (dataframe): the dataframe containing data
        output_path (string): path to save the figure
        period_num (int): the period length based on dataframe's resolution
        model (string): "additive" or multiplicative

    Returns:
        None
    """
    output_path = output_path + str(period_num) + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    start_time = '2022-01-01 00:00:00'
    end_time = '2022-12-31 23:00:00'
    input_df = input_df.loc[(input_df['Datetime'] >= start_time) & (input_df['Datetime'] <= end_time)]
    
    temp_df = input_df
    temp_df.set_index('Datetime', inplace=True)
    for column in temp_df.columns:
        print(column)
        plot_df = temp_df[[column]]
        check = (plot_df[column] == 0).all()
        if (not check):
            plot_df[column] = plot_df[column].apply(lambda x : x if x > 0 else 0)
            plot_df.replace(to_replace=0, value=1e-6, inplace=True)
            
            # Perform seasonal decomposition
            result = seasonal_decompose(plot_df, model=model, period=period_num)  # Adjust period as needed

            # Create a Matplotlib figure and axes
            fig, axes = plt.subplots(4, 1, figsize=(20, 8))

            # Plot the original time series
            axes[0].plot(plot_df, label='Original', color="#023e8a")
            axes[0].legend()

            # Plot the trend component
            axes[1].plot(result.trend, label='Trend', color="#0077b6")
            axes[1].legend()

            # Plot the seasonal component
            axes[2].plot(result.seasonal, label='Seasonal', color='#03045e')
            axes[2].legend()
        
            # Plot the residual component
            axes[3].plot(result.resid, label='Residual', color='#780000')
            axes[3].legend()
        
            # Adjust layout
            plt.tight_layout()
            plt.savefig(output_path + "seasonality_" + str(period_num) + "_" + column + ".png", dpi=600)
            plt.close()
        else:
            pass
    
    return None

def socio_economic_analysis():
    
    
    return None


def weather_analysis():
    
    
    
    return None
    