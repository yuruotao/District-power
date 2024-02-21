import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

# Import modules
import script.analysis as analysis
import script.imputation as imputation
import script.missing_value as missing_value
import script.profiling as profiling
import script.basic_statistics as basic_statistics
import script.resample as resample
import script.uniform as uniform


def time_series_dataframe_create(start_time, stop_time, time_interval):
    """create a dataframe with a column containing a time series, which is the return value

    Args:
        start_time (string): the start time of the time series
        stop_time (string): the stop time of the time series
        time_interval (string): the time interval of the time series

    Returns:
        _datafra_: dataframe containing a column of time series
    """

    time_index = pd.date_range(start=start_time, end=stop_time, freq=time_interval)
    # Create a DataFrame with the time series column
    time_series_df = pd.DataFrame({'Datetime': time_index})
    return time_series_df

def separate_by(input_df, meta_df, output_path):
    """separate the dataframe by city, into individual xlsx files

    Args:
        input_df (dataframe): the input dataframe to be separated
        meta_df (dataframe): the dataframe containing information about input_df
        output_path (string): the folder to contain the output xlsx files

    Returns:
        None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    
    
    
    
    return None

def district_aggregate(input_df, level, output_path):
    """aggregate the data by different levels

    Args:
        input_df (dataframe): the input dataframe containing data
        level (int): level for aggregation, 1 for city level, 2 for district level, 0 for all
        output_path (string): the output directory to save the intermediate file

    Returns:
        dataframe: the aggregated dataframe
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    datetime_column = input_df["Datetime"]
    input_df = input_df.drop(columns=["Datetime"])
    
    if level == 2:
        name = "district"
        # Step 1: Extract common prefix
        common_prefix = input_df.columns.str.split('-').str[:2].str.join('-')
        # Step 2 and 3: Group and sum
        df_grouped = input_df.groupby(common_prefix, axis=1).sum()
        # Step 4: Rename columns
        df_grouped.columns = common_prefix.unique()
        df_grouped = df_grouped.reindex(sorted(df_grouped.columns), axis=1)
        output_df = df_grouped
        
    elif level == 1:
        name = "city"
        # Step 1: Extract 'a' part from column names
        a_part = input_df.columns.str.split('-').str[0]
        # Step 2 and 3: Group and sum
        df_grouped = input_df.groupby(a_part, axis=1).sum()
        # Step 4: Rename columns (optional, if you want)
        df_grouped.columns = a_part.unique()
        df_grouped = df_grouped.reindex(sorted(df_grouped.columns), axis=1)
        output_df = df_grouped
    
    elif level == 0:
        name = "all"
        output_df = input_df.sum(axis=1)
        output_df = output_df.to_frame()
        output_df.rename(columns={output_df.columns[0]: "Power" }, inplace = True)
        
    output_df = pd.concat([datetime_column, output_df], axis=1)
    output_df.to_excel(output_path + name + ".xlsx", index=False)
    
    return output_df

if __name__ == "__main__":
    
    imputed_df = pd.read_excel("./result/imputation/imputed_data_Forward-Backward.xlsx")
    
    district_df = district_aggregate(imputed_df, 2, "./result/aggregate/")
    city_df = district_aggregate(imputed_df, 1,"./result/aggregate/")
    
    
    