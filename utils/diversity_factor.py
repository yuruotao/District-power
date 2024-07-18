# Coding: utf-8
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib.dates as mdates
import calplot

sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
sns.set_theme(style="white")
mpl.rcParams['font.family'] = 'Times New Roman'

def diversity_factor_all(input_df, meta_df, type):
    """calculate the diversity factor for all transformers

    Args:
        input_df (dataframe): the dataframe containing all transformers' load profile
        meta_df (dataframe): the dataframe containing rated capacity for transformers
        type (string): specify the definition of max_load. "Rated_capacity" for meta.xlsx reference

    Returns:
        dataframe: containing the DF and DATETIME

    """
    DATETIME_column = input_df["DATETIME"]
    temp_df = input_df.drop(["DATETIME"], axis=1)
    
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
    diversity_factor_df = pd.concat([DATETIME_column, diversity_factor_df], axis=1)
    
    return diversity_factor_df

def diversity_factor(input_df, meta_df, type):
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
    DATETIME_column = input_df["DATETIME"]
    temp_df = input_df.drop(["DATETIME"], axis=1)
    
    # Group columns by the first two parts of the column names
    grouped = temp_df.groupby(temp_df.columns.str.split('-', expand=True).map(lambda x: '-'.join(x[:2])), axis=1)
    # Extract sub-DataFrames with the same 'a' and 'b' values
    sub_dataframes = [sub_df for _, sub_df in grouped]

    name_list = []
    # Print sub-DataFrames
    iter = 0
    for i, sub_df in enumerate(sub_dataframes, 1):
        print(i)
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
        diversity_factor_df = pd.concat([DATETIME_column, diversity_factor_df], axis=1)
        diversity_factor_df["DISTRICT"] = str(city_num) + "-" + str(district_num)

        if iter == 0:
            final_DF_df = diversity_factor_df
            iter += 1
        else:
            final_DF_df = pd.concat([final_DF_df, diversity_factor_df], axis=0)
        
    return final_DF_df

def diversity_heatmap(input_df, name, output_path):
    """plot the heatmap of diversity factor

    Args:
        input_df (dataframe): dataframe of diversity factor
        name (string): specify the save name
        output_path (string): folder to store the heatmap plot

    Returns:
        None
    """
    sns.set_theme(style="white")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    start_time = '2022-01-01 00:00:00'
    end_time = '2023-11-10 08:00:00'

    df = input_df
    df = df.loc[(df['DATETIME'] >= start_time) & (df['DATETIME'] <= end_time)]
    
    df['hour'] = df['DATETIME'].dt.hour
    df['date'] = df['DATETIME'].dt.date
    df_pivot = df.pivot_table(index='date', columns='hour', values='Diversity Factor', aggfunc='mean')

    # Plot heatmap
    plt.rc('legend', fontsize=10.5)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_pivot, cmap="crest", annot=False, ax=ax)
    # Get the current x-axis tick labels and positions
    ax.tick_params(axis='both', which='major', labelsize=10.5)

    xticklabels = ax.get_xticklabels()
    xtickpositions = ax.get_xticks()
    yticklabels = ax.get_yticklabels()
    ytickpositions = ax.get_yticks()

    # Set the step size for displaying xticks (e.g., display every nth tick)
    step_size_x = 1
    filtered_xticklabels = [label.get_text() for i, label in enumerate(xticklabels) if i % step_size_x == 0]
    filtered_xtickpositions = [position for i, position in enumerate(xtickpositions) if i % step_size_x == 0]
    
    step_size_y = 3
    filtered_yticklabels = [label.get_text() for i, label in enumerate(yticklabels) if i % step_size_y == 0]
    filtered_ytickpositions = [position for i, position in enumerate(ytickpositions) if i % step_size_y == 0]

    # Set the filtered xtick labels and positions
    ax.set_xticks(filtered_xtickpositions)
    ax.set_xticklabels(filtered_xticklabels, rotation=0)  
    ax.set_yticks(filtered_ytickpositions)
    ax.set_yticklabels(filtered_yticklabels, rotation=0)        

    ax.set_xlabel('Hour', fontsize=10.5)
    ax.set_ylabel('Date', fontsize=10.5)
    
    cax = ax.figure.axes[-1]
    cax.tick_params(labelsize=10.5)
    
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
    input_df = input_df.set_index('DATETIME').resample('D').sum()
    input_df = input_df.reset_index(drop=False)
    DATETIME_column = input_df["DATETIME"]
    temp_df = input_df.drop(columns=["DATETIME"])
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
    diversity_factor_df = pd.concat([DATETIME_column, diversity_factor_df], axis=1)

    diversity_factor_df = diversity_factor_df.set_index("DATETIME")
    calplot.calplot(diversity_factor_df["Diversity Factor"], cmap="Blues")
    plt.savefig(output_path + "DF_daily_all.png", dpi=600)
    plt.close()
    
    return None 
