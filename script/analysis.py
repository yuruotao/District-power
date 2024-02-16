import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import calplot


def average_load_profile(input_df, output_path):
    """Plot the load profile curve

    Args:
        input_df (dataframe): contain the data to be plotted
        output_path (string): path to save the plot

    Returns:
        None
    """
    
    sns.set_theme(style="whitegrid")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    start_time = '2022-01-01 00:00:00'
    end_time = '2023-11-10 08:00:00'
    temp_df = input_df.drop(["Datetime"], axis=1)
    
    input_df = input_df.loc[(input_df['Datetime'] >= start_time) & (input_df['Datetime'] <= end_time)]
    color_list = ["#669bbc", "#003049", "#780000", 
        "#0466c8", "#d90429",  "#0353a4", "#023e7d", "#002855", "#001845", "#001233", "#33415c", "#5c677d", "#7d8597", "#979dac", ]

    column_names = temp_df.columns
    subplot_num = 3
    fig, axes = plt.subplots(subplot_num, 1, figsize=(20, 7))
    for iter in range(subplot_num):

        sns.lineplot(data=input_df, x='Datetime', y=column_names[iter], ax=axes[iter], color=color_list[iter])
        axes[iter].set_title(column_names[iter])

        n = 1000  # Set the desired frequency of ticks
        ticks = input_df.iloc[::n, 0]  # Select every nth tick from the 'Date' column
        axes[iter].set_xticks(ticks)
        axes[iter].set(xlabel="", ylabel="")

            
    # Hide xticks for all subplots except the bottom one
    for ax in axes[:-1]:
        ax.xaxis.set_tick_params(which='both', bottom=False, top=False, labelbottom=False)

    # Display xticks only at the bottom subplot
    axes[-1].xaxis.set_ticks_position('both')  # Display ticks on both sides
    axes[-1].xaxis.set_tick_params(which='both', bottom=True, top=False)  # Only bottom ticks are visible
    
    
    # Rotate the x-tick labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout
    plt.tight_layout()
    # Show the plot
    plt.savefig(output_path + "city_load_profile.png", dpi=600)
    plt.close()
    
    return None

def diversity_factor_all(input_df, meta_df, output_path, type):
    """calculate the diversity factor for all transformers

    Args:
        input_df (dataframe): the dataframe containing all transformers' load profile
        meta_df (dataframe): the dataframe containing rated capacity for transformers
        output_path (string): path to store the output xlsx
        type (string): specify the definition of max_load. "Rated_capacity" for meta.xlsx reference

    Returns:
        list: containing the DF dataframe of input dataframe
        list: contain the name of dataframes
    """
    datetime_column = input_df["Datetime"]
    temp_df = input_df.drop(["Datetime"], axis=1)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Calculate total load for each hour by summing across all transformers
    total_load_per_hour = temp_df.sum(axis=1)
    print(total_load_per_hour)
    
    if type == "Rated_capacity":
        max_load = meta_df["YXRL"].sum()
        
    else:
        # Calculate maximum load across all hours
        max_load = total_load_per_hour.max()
    
    # Calculate diversity factor for each hour
    diversity_factor = total_load_per_hour / max_load
    diversity_factor_df = pd.DataFrame(diversity_factor, columns=['Diversity Factor'])
    diversity_factor_df = pd.concat([datetime_column, diversity_factor_df], axis=1)
    
    print(diversity_factor_df)
    diversity_factor_df.to_excel(output_path + "DF_all_transformers.xlsx", index=False)
    
    return [diversity_factor_df], ["all"]

def diversity_factor(input_df, meta_df, output_path, type):
    """calculate the diversity factor for all transformers in the same district

    Args:
        input_df (dataframe): the dataframe containing all transformers' load profile
        meta_df (dataframe): the dataframe containing rated capacity for transformers
        output_path (string): path to store the output xlsx
        type (string): specify the definition of max_load. "Rated_capacity" for meta.xlsx reference

    Returns:
        list: containing the DF dataframes for each district
        list: contain the name of dataframes
    """
    datetime_column = input_df["Datetime"]
    temp_df = input_df.drop(["Datetime"], axis=1)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Group columns by the first two parts of the column names
    grouped = temp_df.groupby(temp_df.columns.str.split('-', expand=True).map(lambda x: '-'.join(x[:2])), axis=1)
    # Extract sub-DataFrames with the same 'a' and 'b' values
    sub_dataframes = [sub_df for _, sub_df in grouped]

    return_list = []
    name_list = []
    # Print sub-DataFrames
    for i, sub_df in enumerate(sub_dataframes, 1):
        name = sub_df.columns[0].rsplit('-', 1)[0]
        name_list.append(name)
        print("Sub-DataFrame ", name)
        index = name.split("-")
        city_num = int(index[0])
        district_num = int(index[1])
        
        # Calculate total load for each hour by summing across all transformers
        total_load_per_hour = sub_df.sum(axis=1)
        
        # Calculate maximum load across all hours
        if type == "Rated_capacity":
            meta_df_temp = meta_df.loc[meta_df['City'] == city_num and meta_df['District'] == district_num]
            max_load = meta_df_temp["YXRL"].sum()
        else:
            # Calculate maximum load across all hours
            max_load = total_load_per_hour.max()
        
        # Calculate diversity factor for each hour
        diversity_factor = total_load_per_hour / max_load
        diversity_factor_df = pd.DataFrame(diversity_factor, columns=['Diversity Factor'])
        diversity_factor_df = pd.concat([datetime_column, diversity_factor_df], axis=1)
        return_list.append(diversity_factor_df)
        print(diversity_factor_df)
        diversity_factor_df.to_excel(output_path + "DF_" + name + ".xlsx", index=False)
    
    return return_list, name_list

def diversity_heatmap(input_df_list, name_list, output_path):
    """plot the heatmap of diversity factor

    Args:
        input_df_list (list): list containing dataframes of diversity factor
        name_list (list): contain the names of dataframes
        output_path (string): folder to store the heatmap plot

    Returns:
        None
    """
    sns.set_theme(style="whitegrid")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    start_time = '2022-01-01 00:00:00'
    end_time = '2023-11-10 08:00:00'
        
    for iter in range(len(input_df_list)):
        df = input_df_list[iter]
        name = name_list[iter]
        df = df.loc[(df['Datetime'] >= start_time) & (df['Datetime'] <= end_time)]
        
        df['hour'] = df['Datetime'].dt.hour
        df['date'] = df['Datetime'].dt.date
        df_pivot = df.pivot_table(index='date', columns='hour', values='Diversity Factor', aggfunc='mean')

        # Plot heatmap
        ax = sns.heatmap(df_pivot, cmap="crest", annot=False)
        # Get the current x-axis tick labels and positions
        xticklabels = ax.get_xticklabels()
        xtickpositions = ax.get_xticks()

        # Set the step size for displaying xticks (e.g., display every nth tick)
        step_size = 2

        # Filter the xtick labels and positions to show only every step_size-th tick
        filtered_xticklabels = [label.get_text() for i, label in enumerate(xticklabels) if i % step_size == 0]
        filtered_xtickpositions = [position for i, position in enumerate(xtickpositions) if i % step_size == 0]

        # Set the filtered xtick labels and positions
        ax.set_xticks(filtered_xtickpositions)
        ax.set_xticklabels(filtered_xticklabels, rotation=0)        
        ax.set(xlabel="", ylabel="")
        
        plt.tight_layout()
        plt.savefig(output_path + "DF_" + name + ".png", dpi=600)
        plt.close()
    
    return None

def year_DF_heatmap(input_df, meta_df, output_path, type):
    """Plot the Github style heatmap for diversity factor
    https://python.plainenglish.io/interactive-calendar-heatmaps-with-plotly-the-easieast-way-youll-find-5fc322125db7

    Args:
        input_df (dataframe): the dataframe containing all transformers' load profile
        meta_df (dataframe): the dataframe containing rated capacity for transformers
        output_path (string): path to store the output xlsx
        type (string): specify the definition of max_load. "Rated_capacity" for meta.xlsx reference

    Returns:
        None
    """
    datetime_column = input_df["Datetime"]
    temp_df = input_df.drop(["Datetime"], axis=1)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Calculate total load for each day by summing across all transformers
    total_load_per_day = temp_df.sum(axis=1)
    
    if type == "Rated_capacity":
        max_load = 24 * meta_df["YXRL"].sum()
    else:
        # Calculate maximum load across all hours
        max_load = total_load_per_day.max()

    # Calculate diversity factor for each day
    diversity_factor = total_load_per_day / max_load
    diversity_factor_df = pd.DataFrame(diversity_factor, columns=['Diversity Factor'])
    diversity_factor_df.reset_index()
    diversity_factor_df = pd.concat([datetime_column, diversity_factor_df], axis=1)

    diversity_factor_df.to_excel(output_path + "DF_daily_all_transformers.xlsx", index=False)
    diversity_factor_df = diversity_factor_df.set_index("Datetime")
    calplot.calplot(diversity_factor_df["Diversity Factor"], cmap="Blues")
    plt.savefig(output_path + "DF_daily_all.png", dpi=600)
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

def weather_analysis():
    
    
    
    return None

def extreme_weather_detect(input_df, output_path):
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    return None

def extreme_weather_plot(input_df, city, weather_data_path, start_time, end_time, output_path):
    sns.set_theme(style="whitegrid")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    weather_df = pd.read_excel(weather_data_path)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    time_index = pd.date_range(start=start_time, end=end_time, freq="h")
    # Create a DataFrame with the time series column
    time_series_df = pd.DataFrame({'Datetime': time_index})
    weather_df = pd.merge(time_series_df, weather_df, on='Datetime', how="left")
    # Filter out missing values from weather_df
    weather_df_filtered = weather_df.fillna("None")
    
    time_series_df = pd.merge(time_series_df, input_df, on='Datetime', how="left")
    time_series_df = pd.merge(time_series_df, weather_df_filtered, on='Datetime', how="left")
    time_series_df = time_series_df.set_index("Datetime")
    
    event_colors = {'Hot weather': '#bc4749', 
                    'Severe convective weather': '#4f000b', 
                    'Cold wave': '#5a189a',
                    'Dragon-boat rain': '#00a6fb',
                    'Drought': '#e09f3e',
                    'Excessive flooding': '#003554',
                    'Extreme rainfall': '#0582ca',
                    'Rainstorm': '#006494',
                    'Tropical Storm Sanba': '#ff6700',
                    'Typhoon Chaba': '#4f772d',
                    'Typhoon Haikui': '#4f772d',
                    'None':"#FFFFFF"}
    

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(time_series_df.index, time_series_df['Power'], color='#274c77')

    # Set background color according to Event values
    for event, color in event_colors.items():
        subset = time_series_df[time_series_df['Event'] == event]
        print(event)
        
        dfs = []
        start_idx = subset.index[0]
        for idx in subset.index:
            if (idx - start_idx).days == 1:
                dfs.append(subset.loc[start_idx:idx])
                start_idx = idx + pd.Timedelta(hours=1)
        
        for group_df in dfs:
            if event == "None":
                ax.axvspan(group_df.index[0], group_df.index[-1], facecolor=color, alpha=0)
            else:
                ax.axvspan(group_df.index[0], group_df.index[-1], facecolor=color, alpha=0.3)
        
    
    legend = plt.legend()
    legend.get_frame().set_facecolor('none')
    plt.legend(frameon=False)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)        
    ax.set(xlabel="", ylabel="")
    
    plt.tight_layout()
    plt.savefig(output_path + "extreme_weather_" + city + ".png", dpi=600)
    plt.close()
    
    return None

