# Coding: utf-8
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose

sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
sns.set_theme(style="white")
mpl.rcParams['font.family'] = 'Times New Roman'

def average_load_profiles(input_df, output_path):
    """Plot the load profile curve for each column

    Args:
        input_df (dataframe): contain the data to be plotted
        output_path (string): path to save the plot

    Returns:
        None
    """
    
    sns.set_theme(style="white")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    start_time = '2022-01-01 00:00:00'
    end_time = '2023-11-10 08:00:00'

    alphabet_list = [chr(chNum) for chNum in list(range(ord('a'),ord('z')+1))]
    input_df = input_df.loc[(input_df['DATETIME'] >= start_time) & (input_df['DATETIME'] <= end_time)]

    # Create a figure and axes
    fig, axs = plt.subplots(4, 3, figsize=(12, 10), sharey=False, sharex=False)

    # Flatten the axes array for easy iteration
    axs = axs.flatten()

    mpl.rc('xtick', labelsize=10.5)
    mpl.rc('ytick', labelsize=10.5)
    plt.rc('legend', fontsize=10.5)
    
    # Plot each column
    for i, column in enumerate(input_df.columns[1:]):  # Exclude the DATETIME column
        ax = axs[i]
        ax.plot(input_df['DATETIME'], input_df[column], color="#0466c8", linewidth=1.5)
        ax.set_title("(" + alphabet_list[i] + ") City " + column, fontsize=10.5)
        ax.set_xlim(input_df['DATETIME'].min(), input_df['DATETIME'].max())
        ax.tick_params(axis='both', which='major', labelsize=10.5)
        # Set xticks
        xticklabels = ax.get_xticklabels()
        xtickpositions = ax.get_xticks()
        step_size_x = 2
        filtered_xticklabels = [label.get_text() for i, label in enumerate(xticklabels) if i % step_size_x == 0]
        filtered_xtickpositions = [position for i, position in enumerate(xtickpositions) if i % step_size_x == 0]
        ax.set_xticks(filtered_xtickpositions)
        ax.set_xticklabels(filtered_xticklabels, rotation=45)  
        ax.grid(False)

    # Hide the empty subplots
    for ax in axs[len(input_df.columns)-1:]:
        ax.axis('off')

    # Adjust layout
    plt.tight_layout(rect=[0.02, 0.05, 1, 1])
    # Add ylabel to the entire figure
    fig.text(0.004, 0.5, 'Power (kW)', va='center', rotation='vertical', fontsize=10.5)
    
    plt.savefig(output_path + "city_load_profile.png", dpi=600)
    plt.close()

    return None

def specific_load_profile_plot(input_df, start_time, end_time, start_time_1, end_time_1, time_type, output_path):
    """Plot load profile for 2 time intervals

    Args:
        input_df (dataframe): dataframe containing data to be plotted
        start_time (string): start time for time interval 0
        end_time (string): end time for time interval 0
        start_time_1 (string): start time for time interval 1
        end_time_1 (string): end time for time interval 1
        time_type (string): time interval range
        output_path (string): path to save the plot

    Returns:
        None
    """
    
    sns.set_theme(style="white")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    alphabet_list = [chr(chNum) for chNum in list(range(ord('a'),ord('z')+1))]
    
    input_df_0 = input_df.loc[(input_df['DATETIME'] >= pd.to_datetime(start_time)) & (input_df['DATETIME'] <= pd.to_datetime(end_time))]
    input_df_1 = input_df.loc[(input_df['DATETIME'] >= pd.to_datetime(start_time_1)) & (input_df['DATETIME'] <= pd.to_datetime(end_time_1))]
    
    if time_type == "Day":
        time_index = pd.date_range(start="00:00:00", end="23:00:00", freq='h')
        label_0 = "Monday"
        label_1 = "Sunday"
        
    elif time_type == "Week":
        time_index = pd.date_range(start="2022-07-01 00:00:00", end="2022-07-07 23:00:00", freq='h')
        label_0 = "July 4-July 10"
        label_1 = "July 11-July 17"
        
    elif time_type == "Month":
        time_index = pd.date_range(start="2022-07-01 00:00:00", end="2022-07-31 23:00:00", freq='h')
        label_0 = "July"
        label_1 = "August"
        
    input_df_0["DATETIME"] = time_index
    input_df_1["DATETIME"] = time_index
    
    # Create a figure and axes
    fig, axs = plt.subplots(2, 3, figsize=(14, 10))

    # Flatten the axes array for easy iteration
    axs = axs.flatten()
    
    mpl.rc('xtick', labelsize=10.5)
    mpl.rc('ytick', labelsize=10.5)
    plt.rc('legend', fontsize=10.5)

    # Plot each column
    alphabet_num = 0
    for i, column in enumerate(input_df.columns[1:]):  # Exclude the datetime column
        ax = axs[i]
        ax.tick_params(axis='both', which='major', labelsize=10.5)
        if alphabet_num == 0:
            ax.plot(input_df_0['DATETIME'], input_df_0[column], color="#0466c8", label=label_0, linewidth=1.5)
            ax.plot(input_df_1['DATETIME'], input_df_1[column], color="#d90429", label=label_1, linewidth=1.5)
        else:
            ax.plot(input_df_0['DATETIME'], input_df_0[column], color="#0466c8", linewidth=1.5)
            ax.plot(input_df_1['DATETIME'], input_df_1[column], color="#d90429", linewidth=1.5)
        ax.set_title("(" + alphabet_list[alphabet_num] + ") City " + column, fontsize=10.5)
        ax.set_xlim(input_df_0['DATETIME'].min(), input_df_0['DATETIME'].max())
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d %H:%M'))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)   
        ax.grid(False)
        alphabet_num = alphabet_num + 1

    # Hide the empty subplots
    for ax in axs[len(input_df.columns)-1:]:
        ax.axis('off')

    if time_type == "Day":
        fig.legend(loc='lower right', fontsize=10.5, bbox_to_anchor=(0.8, 0.2))
    elif time_type == "Week":
        fig.legend(loc='lower right', fontsize=10.5, bbox_to_anchor=(0.85, 0.2))
    elif time_type == "Month":
        fig.legend(loc='lower right', fontsize=10.5, bbox_to_anchor=(0.8, 0.2))
    
    # Adjust layout
    plt.tight_layout(rect=[0.02, 0.05, 1, 1])
    # Add ylabel to the entire figure
    fig.text(0.004, 0.5, 'Power (kW)', va='center', rotation='vertical', fontsize=10.5)
    # Show the plot
    plt.savefig(output_path + "city_load_profile_" + time_type + ".png", dpi=600)
    plt.close()
    
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
    
    sns.set_theme(style="white")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    output_path = output_path + str(period_num) + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    start_time = '2022-01-01 00:00:00'
    end_time = '2022-12-31 23:00:00'
    input_df = input_df.loc[(input_df['DATETIME'] >= start_time) & (input_df['DATETIME'] <= end_time)]
    
    temp_df = input_df
    temp_df.set_index('DATETIME', inplace=True)
    
    mpl.rc('xtick', labelsize=10.5)
    mpl.rc('ytick', labelsize=10.5)
    plt.rc('legend', fontsize=10.5)
    
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
            fig, axes = plt.subplots(4, 1, figsize=(14, 10), layout='compressed')

            # Plot the original time series
            axes[0].plot(plot_df, label='Original', color="#023e8a")
            axes[0].set_xlim(plot_df.index.min(), plot_df.index.max())
            axes[0].legend()

            # Plot the trend component
            axes[1].plot(result.trend, label='Trend', color="#0077b6")
            axes[1].set_xlim(plot_df.index.min(), plot_df.index.max())
            axes[1].legend()

            # Plot the seasonal component
            axes[2].plot(result.seasonal, label='Seasonal', color='#03045e')
            axes[2].set_xlim(plot_df.index.min(), plot_df.index.max())
            axes[2].legend()
        
            # Plot the residual component
            axes[3].plot(result.resid, label='Residual', color='#780000')
            axes[3].set_xlim(plot_df.index.min(), plot_df.index.max())
            axes[3].legend()

            plt.xlabel("Time", fontsize=10.5)
            plt.tight_layout(rect=[0.02, 0.05, 1, 1])
            # Add ylabel to the entire figure
            fig.text(0.004, 0.5, 'Power (kW)', va='center', rotation='vertical', fontsize=10.5)
            plt.savefig(output_path + model + "_seasonality_" + str(period_num) + "_" + column + ".png", dpi=600)
            plt.close()
        else:
            pass
    
    return None
