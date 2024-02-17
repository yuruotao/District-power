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
    # Setup
    # Import raw data and meta data
    raw_data_path = "./data/raw_data.xlsx"
    meta_path = "./data/meta.xlsx"
    
    raw_data_df = pd.read_excel(raw_data_path)
    meta_df = pd.read_excel(meta_path)
    
    # Define the time interval
    start_time = '2022-01-01 00:00:00'
    end_time = '2023-11-10 08:00:00'

    # Create a new DataFrame based on the time interval
    #raw_data_df = raw_data_df.loc[(raw_data_df['Datetime'] >= start_time) & (raw_data_df['Datetime'] <= end_time)]
    #raw_data_df = raw_data_df.reset_index()
    
    ####################################################################################################
    # Missing value
    """
    missing_value.missing_value_visualization(raw_data_df, "./result/missing_value")
    """
    ####################################################################################################
    # Profiling
    """
    profiling.profiling(raw_data_df, "./result/profile/raw", "json")
    profiling.profiling(raw_data_df, "./result/profile/raw", "html")
    basic_statistics.basic_statistics(raw_data_df, "./result/basic_statistics/raw")
    """
    ####################################################################################################
    # Adjust by deleting (Manually)
    """
    # Delete the columns whose missing value takes up more than 20%
    raw_data_adjusted_df = pd.read_excel("./data/raw_data_adjusted.xlsx")
    raw_data_adjusted_df = raw_data_adjusted_df.loc[(raw_data_adjusted_df['Datetime'] >= start_time) & (raw_data_adjusted_df['Datetime'] <= end_time)]
    raw_data_adjusted_df = raw_data_adjusted_df.reset_index()
    basic_statistics.basic_statistics(raw_data_adjusted_df, "./result/basic_statistics/adjusted")
    """
    ####################################################################################################
    # Imputation
    """
    #imputation_methods = ["Linear", "Forward", "Backward", "Forward-Backward", "Average", "MICE", "BiScaler", "AutoML"]
    #for method in imputation_methods:
        #imputed_df = imputation.imputation(raw_data_adjusted_df, save_path="./result/imputation", imputation_method=method)
        #imputed_df = pd.read_excel("./result/imputation/imputed_data_" + method + ".xlsx")
        #basic_statistics.basic_statistics(imputed_df, "./result/basic_statistics/imputation/" + method)
    
    #imputed_df = imputation.imputation(raw_data_adjusted_df, save_path="./result/imputation", imputation_method="BiScaler")
    #imputed_df = pd.read_excel("./result/imputation/imputed_data_BiScaler.xlsx")
    #basic_statistics.basic_statistics(imputed_df, "./result/basic_statistics/imputation/BiScaler")
    
    #imputation.imputation_visualization(raw_data_df, '2022-01-01 00:00:00', '2022-01-08 00:00:00', 
    #                                    ["Linear", "Forward", "Backward", "Forward-Backward"],
    #                                    "0-0-0",
    #                                    "./result/imputation/")
    """
    imputed_df = pd.read_excel("./result/imputation/imputed_data_Forward-Backward.xlsx")
    ####################################################################################################
    # Resample
    """
    resample_df_list = resample.resample(imputed_df, output_path="./result/resample", freq_list=['6h','D'])
    resample.resample_visualization(imputed_df, resample_df_list, "./result/resample/figure/")
    """
    ####################################################################################################
    # Seasonality decomposition
    """
    district_df = district_aggregate(imputed_df, 2, "./result/aggregate/")
    city_df = district_aggregate(imputed_df, 1,"./result/aggregate/")

    analysis.seasonality_decomposition(imputed_df, "./result/seasonality/additive/", 24, "additive")
    analysis.seasonality_decomposition(imputed_df, "./result/seasonality/additive/", 168, "additive")
    
    analysis.seasonality_decomposition(imputed_df, "./result/seasonality/multiplicative/", 24, "multiplicative")
    analysis.seasonality_decomposition(imputed_df, "./result/seasonality/multiplicative/", 168, "multiplicative")
    """
    ####################################################################################################
    # Diversity factor
    """
    DF_all_list, name_list = analysis.diversity_factor_all(imputed_df, meta_df, "./result/diversity_factor/", "")
    analysis.diversity_heatmap(DF_all_list, name_list, "./result/diversity_factor/")
    
    DF_district_list, name_list = analysis.diversity_factor(imputed_df, meta_df, "./result/diversity_factor/districts/", "")
    analysis.diversity_heatmap(DF_district_list, name_list, "./result/diversity_factor/districts/figure/")
    daily_df = pd.read_excel("./result/resample/resampled_D.xlsx")
    analysis.year_DF_heatmap(daily_df, meta_df, "./result/diversity_factor/", "")
    """
    ####################################################################################################
    # Load profile
    #analysis.average_load_profile(city_df, "./result/load_profile/")
    
    ####################################################################################################
    # Transformer by capacity
    """
    capacity_threashold = 3000
    meta_df = meta_df.astype({"YXRL":int})
    
    valid_meta = meta_df[meta_df['Delete'].isna()]
    valid_meta = valid_meta.drop(columns=["Delete"])
    low_meta = valid_meta[valid_meta["YXRL"] < capacity_threashold]
    low_meta = low_meta.reset_index(drop=True)
    low_strings = []
    # Iterate through the rows of df1
    for index, row in low_meta.iterrows():
        # Concatenate values of columns 'a', 'b', and 'c' into a single string in the "a-b-c" format
        formatted_string = f"{row['City']}-{row['District']}-{row['Transformer']}"
        # Append the formatted string to the list
        low_strings.append(formatted_string)
    low_capacity_df = pd.DataFrame(columns=['Datetime', 'Low Capacity'])
    low_capacity_df['Datetime'] = imputed_df['Datetime']
    low_capacity_df['Low Capacity'] = imputed_df[low_strings].sum(axis=1)
    
    high_meta = valid_meta[valid_meta["YXRL"] >= capacity_threashold]
    high_meta = high_meta.reset_index(drop=True)
    high_strings = []
    # Iterate through the rows of df1
    for index, row in high_meta.iterrows():
        # Concatenate values of columns 'a', 'b', and 'c' into a single string in the "a-b-c" format
        formatted_string = f"{row['City']}-{row['District']}-{row['Transformer']}"
        # Append the formatted string to the list
        high_strings.append(formatted_string)
    high_capacity_df = pd.DataFrame(columns=['Datetime', 'High Capacity'])
    high_capacity_df['Datetime'] = imputed_df['Datetime']
    high_capacity_df['High Capacity'] = imputed_df[high_strings].sum(axis=1)
    
    capacity_df = pd.merge(high_capacity_df, low_capacity_df, on='Datetime', how="left")
    analysis.capacity_plot(capacity_df, "./result/capacity/")
    """
    ####################################################################################################
    # Holiday
    """
    province_df = district_aggregate(imputed_df, 0,"./result/aggregate/")
    analysis.holiday_plot(province_df, "all", "./data/festival.xlsx", start_time, end_time, "./result/festival/")
    """
    ####################################################################################################
    # Weather analysis
    station_set = set(meta_df["Closest_Station"])
    xlsx_base = "./result/NCDC_weather_data/stations_imputed/"
    
    # Weather basic statistics
    """
    for element in station_set:
        temp_xlsx_path = xlsx_base + str(element) + ".xlsx"
        temp_weather_df = pd.read_excel(temp_xlsx_path)
        temp_weather_df = temp_weather_df[["DATE", "RH", "TEMP", "DEWP", "SLP", "STP", "VISIB", "WDSP", "MXSPD", "GUST", "MAX", "MIN", "PRCP", "SNDP"]]
        temp_weather_df = temp_weather_df.rename(columns={"DATE": "Datetime"})
        
        city_df = meta_df.loc[meta_df['Closest_Station'] == element]
        city_num = set(city_df["City"]).pop()
        print("City", city_num)
        
        basic_statistics.basic_statistics(temp_weather_df, "./result/extreme_weather/basic_statistics/city_" + str(city_num))
    """
    # Weather correlation
    """
    for element in station_set:
        temp_xlsx_path = xlsx_base + str(element) + ".xlsx"
        temp_weather_df = pd.read_excel(temp_xlsx_path)
        temp_weather_df = temp_weather_df[["DATE", "RH", "TEMP", "DEWP", "SLP", "STP", "VISIB", "WDSP", "MXSPD", "MAX", "MIN", "PRCP"]]
        temp_weather_df = temp_weather_df.rename({"DATE":"Datetime"}, axis=1)
        temp_weather_df['Datetime'] = pd.to_datetime(temp_weather_df['Datetime'])
        city_df = meta_df.loc[meta_df['Closest_Station'] == element]
        city_num = set(city_df["City"]).pop()
        print("City", city_num)
        
        temp_city_df = district_aggregate(imputed_df, 1,"./result/aggregate/")
        temp_city_df = temp_city_df[["Datetime", str(city_num)]]
        temp_city_df = temp_city_df.rename({str(city_num):"POWER"}, axis=1)
        temp_city_df['Datetime'] = temp_city_df['Datetime'].dt.floor('D')
        temp_city_df['Datetime'] = pd.to_datetime(temp_city_df['Datetime'])
        # Group by day and aggregate values
        temp_city_df = temp_city_df.groupby('Datetime').agg({"POWER": 'max'}).reset_index()
        temp_weather_df = pd.merge(temp_weather_df, temp_city_df, on='Datetime', how="left")
        temp_weather_df = temp_weather_df[temp_weather_df['POWER'].notna()]
        temp_weather_df = temp_weather_df.drop(["Datetime"], axis=1)
        
        analysis.weather_correlation(temp_weather_df, "./result/extreme_weather/correlation/", str(city_num))
    """
    
    # Extreme weather detection
    #analysis.extreme_weather_detect(meta_df, "./result/extreme_weather/city/", start_time, end_time)

    # Extreme weather plot
    # All districts
    #province_df = district_aggregate(imputed_df, 0,"./result/aggregate/")
    #analysis.extreme_weather_plot(province_df, "all", "./data/extreme_weather.xlsx", start_time, end_time, "./result/extreme_weather/")

    # Cities

    city_df = district_aggregate(imputed_df, 1,"./result/aggregate/")
    datetime_col = "Datetime"
    temp_city_df = city_df.drop([datetime_col], axis=1)
    city_list = list(temp_city_df.columns.values)
    for city in city_list:
        iter_df = city_df[["Datetime", city]]
        iter_df = iter_df.rename(columns={city: "Power"})
        analysis.extreme_weather_city_plot(iter_df, city, 
                                      "./result/extreme_weather/city/extreme_weather_" + str(city) + ".xlsx", 
                                      start_time, end_time, 
                                      "./result/extreme_weather/extreme_plot/")
  