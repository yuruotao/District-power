import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import calplot

#sns.set_theme(style="whitegrid")
#sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})


def average_load_profile():
    
    
    
    
    
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
    
    return [diversity_factor_df]

def diversity_factor(input_df, meta_df, output_path, type):
    """calculate the diversity factor for all transformers in the same district

    Args:
        input_df (dataframe): the dataframe containing all transformers' load profile
        meta_df (dataframe): the dataframe containing rated capacity for transformers
        output_path (string): path to store the output xlsx
        type (string): specify the definition of max_load. "Rated_capacity" for meta.xlsx reference

    Returns:
        list: containing the DF dataframes for each district
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
    # Print sub-DataFrames
    for i, sub_df in enumerate(sub_dataframes, 1):
        name = sub_df.columns[0].rsplit('-', 1)[0]
        print("Sub-DataFrame ", name)
        index = name.split("-")
        city_num = int(index[0])
        district_num = int(index[1])
        
        # Calculate total load for each hour by summing across all transformers
        total_load_per_hour = sub_df.sum(axis=0)
        
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
    
    return return_list

def diversity_heatmap(input_df_list, output_path):
    """plot the heatmap of diversity factor

    Args:
        input_df_list (list): list containing dataframes of diversity factor
        output_path (string): folder to store the heatmap plot

    Returns:
        None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    start_time = '2022-01-01 00:00:00'
    end_time = '2022-01-31 23:00:00'
    
        
    for df in input_df_list:
        name = df.columns[1].rsplit('-', 1)[0]
        df = df.loc[(df['Datetime'] >= start_time) & (df['Datetime'] <= end_time)]
        
        df['hour'] = df['Datetime'].dt.hour
        df['date'] = df['Datetime'].dt.date
        df_pivot = df.pivot_table(index='date', columns='hour', values='Diversity Factor', aggfunc='mean')

        # Plot heatmap
        f, ax = plt.subplots()
        htmap = sns.heatmap(df_pivot, cmap="crest", annot=True, ax=ax)
        htmap.set(xlabel=None)
        htmap.set(ylabel=None)
        
        plt.tight_layout()
        plt.savefig(output_path + "DF_" + name + ".png", dpi=600)
        plt.close()
    
    return None

def year_DF_heatmap(input_df, meta_df, output_path, type):
    
    datetime_column = input_df["Datetime"]
    temp_df = input_df.drop(["Datetime"], axis=1)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Calculate total load for each day by summing across all transformers
    total_load_per_day = temp_df.sum(axis=0)
    
    if type == "Rated_capacity":
        max_load = 24 * meta_df["YXRL"].sum()
    else:
        # Calculate maximum load across all hours
        max_load = total_load_per_day.max()

    # Calculate diversity factor for each day
    diversity_factor = total_load_per_day / max_load
    diversity_factor_df = pd.DataFrame(diversity_factor, columns=['Diversity Factor'])
    diversity_factor_df = pd.concat([datetime_column, diversity_factor_df], axis=1)
    
    print(diversity_factor_df)
    diversity_factor_df.to_excel(output_path + "DF_daily_all_transformers.xlsx", index=False)
    
    diversity_factor_df = diversity_factor_df.set_index("Datetime")
    calplot.calplot(diversity_factor_df["Diversity Factor"])
    plt.show()
    
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
    