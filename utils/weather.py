

import pandas as pd
import numpy as np

def weather_missing_filter(meta_df, merged_df, threshold):
    """Delete the stations whose missing data percentage reach the threshold

    Args:
        meta_df (dataframe): dataframe containing the NCDC station meta data
        merged_df (merged_df): raw data merged_df
        threshold (float): threshold for deletion

    Returns:
        None
    """

    
    # Calculate percentage of missing values in each column
    missing_percentages = merged_df.isna().mean() * 100
    
    # Drop columns where the percentage of missing values exceeds the threshold
    columns_to_drop = missing_percentages[missing_percentages > threshold].index
    processed_df = merged_df.drop(columns=columns_to_drop)
    stations_higher_than_threshold = processed_df.columns.to_list()
    stations_higher_than_threshold.remove("DATETIME")
    
    filtered_meta_df = meta_df[meta_df['STATION_ID'].isin(stations_higher_than_threshold)].reset_index(drop=True)

    return filtered_meta_df

def NCDC_weather_data_imputation(filtered_meta_df, merged_df):
    """Reformat and impute the missing data of weather data
    Add relative humidity "RH" to the dataframe

    Args:
        filtered_meta_df (dataframe): meta dataframe
        merged_df (dataframe): dataframe containing merged weather data
        engine (sqlalchemy_engine): engine used for database creation

    Returns:
        None
    """

    station_id_list = filtered_meta_df["STATION_ID"].to_list()
    imputed_df = pd.DataFrame()
    for station_id in station_id_list:
        print(station_id)
        temp_df = merged_df[merged_df["STATION_ID"] == station_id]
        datetime_column = temp_df["DATETIME"]
        temp_df = temp_df.drop(columns=["DATETIME", "ID", "STATION_ID"])

        forward_df = temp_df.shift(-1)
        backward_df = temp_df.shift(1)
        average_values = (forward_df + backward_df) / 2
        
        temp_df = temp_df.copy()
        temp_df[temp_df.isna() & forward_df.notna() & backward_df.notna()] = average_values[temp_df.isna() & forward_df.notna() & backward_df.notna()]
        temp_df[temp_df.isna() & forward_df.notna() & backward_df.isna()] = forward_df[temp_df.isna() & forward_df.notna() & backward_df.isna()]
        temp_df[temp_df.isna() & backward_df.notna() & forward_df.isna()] = backward_df[temp_df.isna() & backward_df.notna() & forward_df.isna()]

        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        temp_df.set_index('DATETIME', inplace=True)
        # Set Datetime column as index
        for column in temp_df.columns:
            mean_value = temp_df[column].mean()
            # Fill NaN values with the mean
            temp_df[column].fillna(mean_value, inplace=True)

        temp_df = temp_df.reset_index()
        temp_df["STATION_ID"] = station_id
        print(temp_df)
        imputed_df = pd.concat([imputed_df, temp_df], axis=0)

    return imputed_df
