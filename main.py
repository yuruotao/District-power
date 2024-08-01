# Coding: utf-8
# Conduct main analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import missingno as msno
import seaborn as sns
import matplotlib as mpl
import matplotlib.dates as mdates
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import time

from utils.load_missing import *
from utils.diversity_factor import *
from utils.load_profile import *
from utils.weather import *

def district_aggregate(input_df, level):
    """aggregate the data by different levels

    Args:
        input_df (dataframe): the input dataframe containing data
        level (int): level for aggregation, 1 for city level, 2 for district level, 0 for all
        
    Returns:
        dataframe: the aggregated dataframe
    """

    datetime_column = input_df['DATETIME']
    input_df = input_df.drop(columns=['DATETIME'])
    
    if level == 2:
        # Step 1: Extract common prefix
        common_prefix = input_df.columns.str.split('-').str[:2].str.join('-')
        # Step 2 and 3: Group and sum
        df_grouped = input_df.groupby(common_prefix, axis=1).sum()
        # Step 4: Rename columns
        df_grouped.columns = common_prefix.unique()
        df_grouped = df_grouped.reindex(sorted(df_grouped.columns), axis=1)
        output_df = df_grouped
        
    elif level == 1:
        # Step 1: Extract 'a' part from column names
        a_part = input_df.columns.str.split('-').str[0]
        # Step 2 and 3: Group and sum
        df_grouped = input_df.groupby(a_part, axis=1).sum()
        # Step 4: Rename columns (optional, if you want)
        df_grouped.columns = a_part.unique()
        df_grouped = df_grouped.reindex(sorted(df_grouped.columns), axis=1)
        output_df = df_grouped
    
    elif level == 0:
        output_df = input_df.sum(axis=1)
        output_df = output_df.to_frame()
        output_df.rename(columns={output_df.columns[0]: "LOAD" }, inplace = True)
        
    output_df = pd.concat([datetime_column, output_df], axis=1)
    
    return output_df

if __name__ == "__main__":
    # 1. Initialization and import data from database
    time_index = pd.date_range(start="2022-01-01 00:00:00", end="2023-11-11 23:00:00", freq="H")
    datetime_df = pd.DataFrame()
    datetime_df['DATETIME'] = time_index
    
    db_address = 'sqlite:///./data/Transformer_DB/Transformer_DB.db'
    engine = create_engine(db_address)

    transformer_raw_query = 'SELECT * FROM transformer_raw'
    transformer_raw_df = pd.read_sql(transformer_raw_query, engine)
    transformer_raw_df = transformer_raw_df.astype({'DATETIME':"datetime64[ms]"})
    
    transformer_meta_query = 'SELECT * FROM transformer_meta'
    transformer_meta_df = pd.read_sql(transformer_meta_query, engine)
    ############################################################################################################
    # 2. Missing data, filter, imputation
    transformer_pivot_df = transformer_raw_df.pivot(index='DATETIME', columns='TRANSFORMER_ID', values='LOAD')
    transformer_pivot_df = datetime_df.merge(transformer_pivot_df, on='DATETIME', how='left')
    # Statistics for load data
    transformer_pivot_df.describe().to_excel("./result/load_summary.xlsx")
    
    # Load missing data
    missing_data_flag = False
    if missing_data_flag:
        load_missing_value_visualization(transformer_pivot_df, "./result/load_missing")

    # Filter by the percentage of missing data
    filtered_transformer_meta_df = transformer_missing_filter(transformer_meta_df, transformer_pivot_df, 30)
    
    # Imputation
    imputed_transformer_df = transformer_data_imputation(filtered_transformer_meta_df, transformer_raw_df)

    # Imputation visualization
    imputation_visualization_flag = False
    if imputation_visualization_flag:
        single_transformer_df = transformer_pivot_df[['DATETIME', "0-0-0"]]
        imputation_methods = ["Linear", "Forward", "Backward", "Forward-Backward"]
        for method in imputation_methods:
            imputed_df = imputation(single_transformer_df, save_path="./result/load_imputation", imputation_method=method, save_flag=True)
            
        imputation_visualization(single_transformer_df, '2022-06-17 00:00:00', '2022-06-19 00:00:00', 
                                            ["Linear", "Forward", "Backward", "Forward-Backward"],
                                            "0-0-0",
                                            "./result/load_imputation/")
    ############################################################################################################
    # 3. Analysis
    # 3.1 Diversity factor plot
    diversity_flag = False
    imputed_transformer_pivot_df = imputed_transformer_df.pivot(index='DATETIME', columns='TRANSFORMER_ID', values='LOAD')
    imputed_transformer_pivot_df = datetime_df.merge(imputed_transformer_pivot_df, on='DATETIME', how='left')
    imputed_transformer_pivot_df = imputed_transformer_pivot_df.astype({'DATETIME':"datetime64[ms]"})
    if diversity_flag:
        DF_df = diversity_factor_all(imputed_transformer_pivot_df, filtered_transformer_meta_df, "")
        diversity_heatmap(DF_df, "all", "./result/diversity_factor/")
        
        extreme_weather_query = 'SELECT * FROM extreme_weather_internet'
        extreme_weather_df = pd.read_sql(extreme_weather_query, engine)
        extreme_weather_df = extreme_weather_df.astype({'DATETIME':"datetime64[ns]"})
        holiday_query = 'SELECT * FROM holiday'
        holiday_df = pd.read_sql(holiday_query, engine)
        holiday_df = holiday_df.astype({'DATETIME':"datetime64[ns]"})
        other_df = pd.merge(extreme_weather_df, holiday_df, on='DATETIME', how='outer')
        other_df['HAZARD'].replace('', pd.NA, inplace=True)
        other_df['HOLIDAY'].replace('', pd.NA, inplace=True)
        other_df['HAZARD'] = other_df['HAZARD'].notna().astype(int)
        other_df['HOLIDAY'] = other_df['HOLIDAY'].notna().astype(int)
    
        year_DF_heatmap(imputed_transformer_pivot_df, filtered_transformer_meta_df, "./result/diversity_factor/", "", other_df)
        # For each district
        sub_DF_plot_flag = False
        if sub_DF_plot_flag:
            district_DF_df = diversity_factor(imputed_transformer_pivot_df, filtered_transformer_meta_df, "")
            district_set = set(district_DF_df["DISTRICT"].to_list())
            for district in district_set:
                temp_DF_df = district_DF_df[district_DF_df["DISTRICT"] == district]
                diversity_heatmap(temp_DF_df, district, "./result/diversity_factor/district/")
    # 3.2 City load profile visualization
    city_profile_flag = False
    province_df = district_aggregate(imputed_transformer_pivot_df, 0)
    city_df = district_aggregate(imputed_transformer_pivot_df, 1)
    district_df = district_aggregate(imputed_transformer_pivot_df, 2)
    if city_profile_flag:
        average_load_profiles(city_df, "./result/load_profile/")
    # 3.3 City load profile in different scales
    select_city_profile_flag = False
    if select_city_profile_flag:
        select_df = district_df[['DATETIME', "0-0", "1-0", "2-0", "3-0", "4-0", "5-0", "6-0", "7-0", "8-0", "9-0"]]
        select_df = select_df.rename(columns={
            "0-0":"0", "1-0":"1", "2-0":"2", 
            "3-0":"3", "4-0":"4", "5-0":"5", 
            "6-0":"6", "7-0":"7", "8-0":"8", 
            "9-0":"9"
        })
        select_df = select_df[['DATETIME', "0", "2", "3", "5", "6", "9"]]
        
        specific_load_profile_plot(select_df, '2022-07-04 00:00:00', '2022-07-04 23:00:00', 
                                            '2022-07-10 00:00:00', '2022-07-10 23:00:00', 
                                            "Day", "./result/load_profile/")
        
        specific_load_profile_plot(select_df, '2022-07-04 00:00:00', '2022-07-10 23:00:00', 
                                            '2022-07-11 00:00:00', '2022-07-17 23:00:00', 
                                            "Week", "./result/load_profile/")

        specific_load_profile_plot(select_df, '2022-07-01 00:00:00', '2022-07-31 23:00:00', 
                                            '2022-08-01 00:00:00', '2022-08-31 23:00:00', 
                                            "Month", "./result/load_profile/")
        
    # 3.4 Seasonality decomposition
    seasonality_flag = False
    if seasonality_flag:
        seasonality_decomposition(city_df, "./result/seasonality/", 24, "additive")
        seasonality_decomposition(city_df, "./result/seasonality/", 168, "additive")

        seasonality_decomposition(city_df, "./result/seasonality/", 24, "multiplicative")
        seasonality_decomposition(city_df, "./result/seasonality/", 168, "multiplicative")
    ############################################################################################################
    # 4. Weather and holidays
    # 4.1 Weather correlation
    time_index = pd.date_range(start="2022-01-01 00:00:00", end="2022-12-31 23:00:00", freq="H")
    datetime_df = pd.DataFrame()
    datetime_df['DATETIME'] = time_index
    
    weather_meta_query = 'SELECT * FROM weather_meta'
    weather_meta_df = pd.read_sql(weather_meta_query, engine)
    
    weather_query = 'SELECT * FROM weather'
    weather_df = pd.read_sql(weather_query, engine)
    weather_df = weather_df.astype({'DATETIME':"datetime64[ns]"})

    weather_correlation_flag = False
    if weather_correlation_flag:
        station_set = set(transformer_meta_df["CLOSEST_STATION"].to_list())
        
        for element in station_set:
            print(element)
            temp_weather_df = weather_df[weather_df["STATION_ID"] == str(element)]
            temp_weather_df = temp_weather_df[["DATETIME", "TEMP", "DEWP"]]
            temp_weather_df['DATETIME'] = pd.to_datetime(temp_weather_df['DATETIME'])
            temp_weather_df = datetime_df.merge(temp_weather_df, on='DATETIME', how='left')
            city = transformer_meta_df.loc[transformer_meta_df['CLOSEST_STATION'] == element]
            city_num = set(city["CITY"]).pop()
            print("CITY", city_num)
            
            temp_city_df = city_df[['DATETIME', str(city_num)]]
            temp_city_df = temp_city_df.rename(columns={str(city_num):"LOAD"})
            
            # Group by day and aggregate values
            temp_weather_df = pd.merge(temp_weather_df, temp_city_df, on='DATETIME', how="left")
            temp_weather_df = temp_weather_df[~temp_weather_df.isin([np.nan, np.inf]).any(axis=1)]
            temp_weather_df = temp_weather_df.drop(['DATETIME'], axis=1)
            
            weather_correlation(temp_weather_df, "./result/weather_correlation/", str(city_num))
    # 4.2 Holidays
    holiday_flag = False
    if holiday_flag:
        holiday_query = 'SELECT * FROM holiday'
        holiday_df = pd.read_sql(holiday_query, engine)
        holiday_df = holiday_df.astype({'DATETIME':"datetime64[ns]"})
        holiday_plot(province_df, "all", holiday_df, "2022-01-01 00:00:00", '2023-11-01 00:00:00', "./result/holiday/")
    
    # 4.3 Extreme weather
    extreme_weather = True
    if extreme_weather:
        # For the whole region
        extreme_weather_query = 'SELECT * FROM extreme_weather_internet'
        extreme_weather_df = pd.read_sql(extreme_weather_query, engine)
        extreme_weather_df = extreme_weather_df.astype({'DATETIME':"datetime64[ns]"})
        #extreme_weather_plot(province_df, "all", extreme_weather_df, "2022-01-01 00:00:00", '2023-11-01 00:00:00', "./result/extreme_weather/")
        
        # For each city
        extreme_weather_calculated_query = 'SELECT * FROM extreme_weather_calculated'
        extreme_weather_calculated_df = pd.read_sql(extreme_weather_calculated_query, engine)
        extreme_weather_calculated_df = extreme_weather_calculated_df.astype({'DATETIME':"datetime64[ns]"})
        extreme_city_flag = False
        if extreme_city_flag:
            for city_num in range(10):
                print("CITY", city_num)
                element_df = transformer_meta_df.loc[transformer_meta_df['CITY'] == city_num]
                element = set(element_df['CLOSEST_STATION']).pop()
                print(element)
                
                temp_city_df = city_df[['DATETIME', str(city_num)]]
                temp_city_df = temp_city_df.rename(columns={str(city_num):"LOAD"})
                temp_city_df = temp_city_df.reset_index(drop=True)
                
                temp_extreme_weather_df = extreme_weather_calculated_df[extreme_weather_calculated_df["STATION_ID"] == str(element)]
                temp_extreme_weather_df['DATETIME'] = pd.to_datetime(temp_extreme_weather_df['DATETIME'])
                if len(temp_extreme_weather_df) != 0:
                    hourly_datetime = pd.date_range(start=temp_extreme_weather_df['DATETIME'].min(), end=temp_extreme_weather_df['DATETIME'].max() + pd.Timedelta(days=1) - pd.Timedelta(hours=1), freq='H')
                    repeated_values = {col: temp_extreme_weather_df[col].repeat(24).reset_index(drop=True) for col in temp_extreme_weather_df.columns if col != 'DATETIME'}
                    temp_extreme_weather_df = pd.DataFrame({
                        'DATETIME': hourly_datetime,
                        **repeated_values})
                    
                    temp_extreme_weather_df = datetime_df.merge(temp_extreme_weather_df, on='DATETIME', how='left')
                    temp_extreme_weather_df = temp_extreme_weather_df.drop(columns=["STATION_ID"])
                    temp_extreme_weather_df = temp_extreme_weather_df.reset_index(drop=True)
                    
                    extreme_weather_city_plot(temp_city_df, str(city_num), 
                                                temp_extreme_weather_df,
                                                "2022-01-01 00:00:00", '2022-12-31 23:00:00', 
                                                "./result/extreme_weather/")
            
        # Comparison plot for one city
        guilin_df = city_df[["DATETIME", "2"]]
        guilin_df = guilin_df.rename(columns={"2": "LOAD"})
        temp_extreme_weather_df = extreme_weather_calculated_df[extreme_weather_calculated_df["STATION_ID"] == "57957099999"]
        temp_extreme_weather_df['DATETIME'] = pd.to_datetime(temp_extreme_weather_df['DATETIME'])
        hourly_datetime = pd.date_range(start=temp_extreme_weather_df['DATETIME'].min(), end=temp_extreme_weather_df['DATETIME'].max() + pd.Timedelta(days=1) - pd.Timedelta(hours=1), freq='H')
        repeated_values = {col: temp_extreme_weather_df[col].repeat(24).reset_index(drop=True) for col in temp_extreme_weather_df.columns if col != 'DATETIME'}
        temp_extreme_weather_df = pd.DataFrame({
            'DATETIME': hourly_datetime,
            **repeated_values})
        
        temp_extreme_weather_df = datetime_df.merge(temp_extreme_weather_df, on='DATETIME', how='left')
        temp_extreme_weather_df = temp_extreme_weather_df.drop(columns=["STATION_ID"])
        temp_extreme_weather_df = temp_extreme_weather_df.reset_index(drop=True)
        
        extreme_normal_comparison_plot(guilin_df,
                                        temp_extreme_weather_df,
                                        "2022-05-01 00:00:00", '2022-12-31 23:00:00', 
                                        "./result/extreme_weather/")
    ############################################################################################################
    