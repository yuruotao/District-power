import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

sns.set_theme(style="whitegrid")
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})


def resample(input_df, output_path, freq_list):
    """resample the data

    Args:
        input_df (dataframe): contain the original data
        output_path (string): the path to save the resampled data
        freq_list (list): store the frequencies to resample

    Returns:
        list: contain the resampled dataframes
    """
    print("Resample begin")
    datetime_column = input_df["Datetime"]
    temp_df = input_df.set_index(["Datetime"])
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    resample_list = []
    for freq in freq_list:
        aggregated_df = temp_df.resample(freq).sum()
        aggregated_df = aggregated_df.reset_index()
        
        aggregated_df.to_excel(output_path + "/resampled_" + freq + ".xlsx", index=False)
        resample_list.append(aggregated_df)
    
    return resample_list

def resample_visualization(input_df, resample_df_list, output_path):
    """visualize the resampled data

    Args:
        input_df (dataframe): the original dataframe
        resample_df_list (list): the list containing resampled dataframes
        column (string): specify the transformer index
        output_path (string): the path to save the figure

    Returns:
        None 
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    start_time = '2022-01-01 00:00:00'
    end_time = '2022-07-31 23:00:00'
    temp_df = input_df.drop(["Datetime"], axis=1)
    
    input_df = input_df.loc[(input_df['Datetime'] >= start_time) & (input_df['Datetime'] <= end_time)]
    resample_df_list[0] = resample_df_list[0].loc[(resample_df_list[0]['Datetime'] >= start_time) & (resample_df_list[0]['Datetime'] <= end_time)]
    resample_df_list[1] = resample_df_list[1].loc[(resample_df_list[1]['Datetime'] >= start_time) & (resample_df_list[1]['Datetime'] <= end_time)]
    
    for column in temp_df.columns:
        # Plotting
        fig, axes = plt.subplots(3, 1, figsize=(20, 7))

        # Plot for DataFrame 1
        sns.lineplot(data=input_df, x='Datetime', y=column, ax=axes[0]).set(xlabel=None)
        axes[0].set_title('1 Hour')
        axes[0].set_xticks([])

        # Plot for DataFrame 2
        sns.lineplot(data=resample_df_list[0], x='Datetime', y=column, ax=axes[1]).set(xlabel=None)
        axes[1].set_title('6 Hours')
        axes[1].set_xticks([])

        # Plot for DataFrame 3
        sns.lineplot(data=resample_df_list[1], x='Datetime', y=column, ax=axes[2]).set(xlabel=None)
        axes[2].set_title('1 Day')
        n = 15  # Set the desired frequency of ticks
        ticks = resample_df_list[1].iloc[::n, 0]  # Select every nth tick from the 'Date' column
        axes[2].set_xticks(ticks)

        # Rotate the x-tick labels for better readability
        plt.xticks(rotation=45)

        # Adjust layout
        plt.tight_layout()
        # Show the plot
        plt.savefig(output_path + "resample_" + column + ".png", dpi=600)
        plt.close()
    
    return None