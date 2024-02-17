import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import gdelt
import requests
from os import listdir
from os.path import isfile, join

def NCDC_weather_data_obtain(meta_path, output_path, start_year, stop_year):
    """Obtain the weather data from NCDC

    Args:
        meta_path (string): xlsx containing the NCDC station data
        output_path (string): folder to contain the weather data
        start_year (int): the start year of data
        stop_year (int): the stop year of data

    Returns:
        None
    """
    # meta_df is obtained from
    # https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/
    meta_df = pd.read_csv(meta_path)
    meta_df = meta_df.loc[meta_df["CTRY"]== "CH"]
    meta_df = meta_df.astype({"BEGIN":int, "END":int})
    meta_df = meta_df.loc[meta_df["END"]>stop_year*10000]
    print(meta_df)
    meta_df.reset_index()
    
    # download the data with respect to "USAF"
    base = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/"
    for year in range(start_year, stop_year):
        base_year = base + str(year) + "/"
        output_year = output_path + str(year) + "/"
        if not os.path.exists(output_year):
            os.makedirs(output_year)
        print(year)
        
        for index, row in meta_df.iterrows():
            usaf_value = row['USAF']
            wban_value = row['WBAN']
            print(usaf_value)
            url = base_year + str(usaf_value) + str(wban_value) + ".csv"
            response = requests.get(url)
            csv_data = response.text
            
            filename = output_year + str(usaf_value) + str(wban_value) + ".csv"
            with open(filename, 'w') as f:
                f.write(csv_data)
    
    return None

def NCDC_weather_data_station_merge(meta_path, 
                                    input_path, output_path, 
                                    start_year, stop_year):
    """Merge the collected data into single station files

    Args:
        meta_path (string): xlsx containing the NCDC station data
        input_path (string): folder to contain the weather data
        output_path (string): folder to save the station weather data
        start_year (int): the start year of data
        stop_year (int): the stop year of data

    Returns:
        None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    meta_df = pd.read_csv(meta_path)
    meta_df = meta_df.loc[meta_df["CTRY"]== "CH"]
    meta_df = meta_df.astype({"BEGIN":int, "END":int, "USAF":str, "WBAN":str})
    meta_df = meta_df.loc[meta_df["END"]>stop_year*10000]
    meta_df.reset_index()
    
    station_str = meta_df["USAF"] + meta_df["WBAN"]
    meta_df["Station_ID"] = station_str
    print(meta_df)
    
    for index, row in meta_df.iterrows():
        Station_ID = str(row["Station_ID"])
        for year in range(start_year, stop_year):
            temp_path = input_path + str(year) + "/" + Station_ID + ".csv"
            temp_df = pd.read_csv(temp_path)
            if year == start_year:
                output_df = temp_df
            else:
                output_df = pd.concat([output_df, temp_df], ignore_index=True)
        output_df.to_excel(output_path + Station_ID + ".xlsx", index=False)
    
    return None

def NCDC_weather_data_imputation(data_path, output_path):
    """Reformat and impute the missing data of weather data
    Add relative humidity "RH" to the dataframe

    Args:
        data_path (string): path to station data
        output_path (string): path to store the imputed data

    Returns:
        None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    station_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    for station_file in station_files:
        print(station_file)
        temp_df = pd.read_excel(data_path + station_file)
        temp_df = temp_df[["STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", 
                           "NAME", "TEMP", "DEWP", "SLP", "STP", "VISIB", 
                           "WDSP", "MXSPD", "GUST", "MAX", "MIN", "PRCP", 
                           "SNDP"]]
        
        # Missing data
        temp_df.replace(99.99, np.nan, inplace=True)
        temp_df.replace(999.9, np.nan, inplace=True)
        temp_df.replace(9999.9, np.nan, inplace=True)
        
        # Degree to Celsius
        temp_df['TEMP'] = temp_df.apply(lambda x: (x['TEMP']-32)*(5/9), axis=1)
        temp_df['MAX'] = temp_df.apply(lambda x: (x['MAX']-32)*(5/9), axis=1)
        temp_df['MIN'] = temp_df.apply(lambda x: (x['MIN']-32)*(5/9), axis=1)
        temp_df['DEWP'] = temp_df.apply(lambda x: (x['DEWP']-32)*(5/9), axis=1)
        
        # Dew point to relative humidity
        def calculate_relative_humidity(dew_point_celsius, air_temperature_celsius):
            # Calculate saturation vapor pressure at dew point and air temperature
            es_td = 6.112 * np.exp(17.67 * dew_point_celsius / (dew_point_celsius + 243.5))
            es_t = 6.112 * np.exp(17.67 * air_temperature_celsius / (air_temperature_celsius + 243.5))

            # Calculate relative humidity
            relative_humidity = 100 * (es_td / es_t)

            return relative_humidity
        
        # RH for relative humidity
        temp_df['RH'] = temp_df.apply(lambda x: calculate_relative_humidity(x['DEWP'], x['TEMP']), axis=1)
        
        # Millibar to kPa
        temp_df['SLP'] = temp_df.apply(lambda x: x['SLP']/10, axis=1)
        temp_df['STP'] = temp_df.apply(lambda x: x['STP']/10, axis=1)
        
        # Miles to km
        temp_df['VISIB'] = temp_df.apply(lambda x: x['VISIB']*1.609, axis=1)
        
        # Knots to m/s
        temp_df['WDSP'] = temp_df.apply(lambda x: x['WDSP']*0.51444, axis=1)
        temp_df['MXSPD'] = temp_df.apply(lambda x: x['MXSPD']*0.51444, axis=1)
        temp_df['GUST'] = temp_df.apply(lambda x: x['GUST']*0.51444, axis=1)
        
        # Inches to meter
        temp_df['PRCP'] = temp_df.apply(lambda x: x['PRCP']*0.0254, axis=1)
        temp_df['SNDP'] = temp_df.apply(lambda x: x['SNDP']*0.0254, axis=1)
        
        
        imputed_df = temp_df.interpolate(method='linear')
        # Select only numeric columns
        numeric_cols = ["TEMP", 'RH', "DEWP", "SLP", "STP", "VISIB", "WDSP", "MXSPD", "GUST", "MAX", "MIN", "PRCP", "SNDP"]
        # Calculate means for numeric columns
        col_means = imputed_df[numeric_cols].mean()
        # Fill NaN values in numeric columns with their respective means
        imputed_df[numeric_cols] = imputed_df[numeric_cols].fillna(col_means)
        
        imputed_df.to_excel(output_path + station_file, index=False)

    return None



if __name__ == "__main__":
    #NCDC_weather_data_obtain("./data/isd-history.csv", "./result/NCDC_weather_data/", 2022, 2023+1)
    
    #NCDC_weather_data_station_merge("./data/isd-history.csv",
    #                                "./result/NCDC_weather_data/", 
    #                                "./result/NCDC_weather_data/stations/",
    #                                2022, 2023+1)
    
    NCDC_weather_data_imputation("./result/NCDC_weather_data/stations/", "./result/NCDC_weather_data/stations_imputed/")