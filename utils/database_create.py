# Coding: utf-8
# Script for creating database
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import gdelt
import requests
from os import listdir
from os.path import isfile, join
import missingno as msno
import seaborn as sns
import matplotlib as mpl
import geopandas as gpd
from shapely.geometry import Point

sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
sns.set_theme(style="white")
mpl.rcParams['font.family'] = 'Times New Roman'


# Weather data
def df_to_gdf(df, lon_name, lat_name):
    """convert dataframe to geodataframe

    Args:
        df (dataframe): input dataframe for conversion
        lon_name (string): column name for longitude
        lat_name (string): column name for latitude

    Returns:
        geodataframe
    """
    geometry = [Point(xy) for xy in zip(df[lon_name], df[lat_name])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    # WGS84 coordinate system
    gdf.set_crs(epsg=4326, inplace=True)

    return gdf

def NCDC_weather_meta_data_obtain(meta_path, start_year, stop_year):
    """Obtain the weather meta data

    Args:
        meta_path (string): xlsx containing the NCDC station data
        start_year (int): the start year of data
        stop_year (int): the stop year of data

    Returns:
        None
    """
    # meta_df is obtained from
    # https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/
    meta_df = pd.read_csv(meta_path)
    meta_df = meta_df.loc[meta_df["CTRY"]== "CH"]
    meta_df = meta_df.astype({"BEGIN":int, "END":int, "USAF":str, "WBAN":str})
    meta_df = meta_df.loc[meta_df["BEGIN"]<=start_year*10000]
    meta_df = meta_df.loc[meta_df["END"]>stop_year*10000]
    meta_df["WBAN"] = meta_df["WBAN"].str.zfill(5)
    station_str = meta_df["USAF"] + meta_df["WBAN"]
    meta_df["Station_ID"] = station_str
    meta_df = meta_df.reset_index(drop=True)

    return meta_df

def NCDC_weather_data_obtain(meta_df, start_year, stop_year):
    
    time_index = pd.date_range(start=str(start_year) + "-01-01", end=str(stop_year) + "-12-31", freq="D")
    temp_time_index = pd.DataFrame()
    temp_time_index["Datetime"] = time_index
    
    base = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/"
    
    for index, row in meta_df.iterrows():
        Station_ID = str(row["Station_ID"])
        print(Station_ID)
        usaf_value = row['USAF']
        wban_value = row['WBAN']
            
        
        for year in range(start_year, stop_year):
            # Obtain data from website
            base_year = base + str(year) + "/"
            url = base_year + str(usaf_value) + str(wban_value) + ".csv"
            response = requests.get(url)
            
            # Convert to dataframe for concatenate
            csv_data = response.text
            temp_list = csv_data.split("\n")
            print(temp_list)
            print(pd.DataFrame(csv_data))
            
            if index == 0 and year == start_year:
                weather_df = temp_df
            else:
                weather_df = pd.concat([weather_df, temp_df], axis=1)
            

    return weather_df

def extreme_weather_detect(weather_meta_df, weather_df, start_date, end_date):
    """Detect the extreme weather based on conditions

    Args:
        input_df (dataframe): contain the weather data
        start_date (string): the start date of extreme weather
        end_date (string): the end date of extreme weather

    Returns:
        None
    """
    
    time_index = pd.date_range(start=start_date, end=end_date, freq='H')
    # Create a DataFrame with the time series column
    datetime_df = pd.DataFrame({'DATETIME': time_index})
        
    station_set = set(input_df["Closest_Station"])
    xlsx_base = "./result/NCDC_weather_data/stations_imputed/"
    
    
    
    for element in station_set:
        temp_xlsx_path = xlsx_base + str(element) + ".xlsx"
        temp_weather_df = pd.read_excel(temp_xlsx_path)
        temp_weather_df['DATETIME'] = pd.to_datetime(temp_weather_df['DATE'], format='%Y-%m-%d').dt.floor('D')
        
        city_df = input_df.loc[input_df['Closest_Station'] == element]
        city_num = next(iter(set(city_df["City"])))
        print("City", city_num)
        extreme_weather_df = datetime_df
        
        # Temperature
        extreme_weather_df['High Temperature'] = np.nan
        MAX_percentile_95 = temp_weather_df['MAX'].quantile(0.95)
        high_temp_df = temp_weather_df.loc[temp_weather_df['MAX'] > MAX_percentile_95]
        for date in high_temp_df['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'High Temperature'] = 1
        print("High Temperature done")
        
        extreme_weather_df['Low Temperature'] = np.nan
        MIN_percentile_5 = temp_weather_df['MIN'].quantile(0.05)
        low_temp_df = temp_weather_df.loc[temp_weather_df['MIN'] < MIN_percentile_5]
        for date in low_temp_df['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Low Temperature'] = 1
        print("Low Temperature done")
        
        # Humidity
        extreme_weather_df['High Humidity'] = np.nan
        RH_percentile_95 = temp_weather_df['RH'].quantile(0.95)
        high_hum_df = temp_weather_df.loc[temp_weather_df['RH'] > RH_percentile_95]
        for date in high_hum_df['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'High Humidity'] = 1
        print("High Humidity done")
        
        # Heat Index
        def calculate_heat_index(temp_celsius, relative_humidity):
            """Constants for the Heat Index calculation
            Input is celsius
            Coefficients are retrieved here
            https://en.wikipedia.org/wiki/Heat_index

            Args:
                temp_celsius (float): _description_
                relative_humidity (float): _description_

            Returns:
                float: 
            """
            temp_fahrenheit = (temp_celsius * 9/5) + 32
            relative_humidity = relative_humidity / 100
    
            # Calculate the Heat Index in Fahrenheit
            HI = (-42.379 + 
                  2.04901523 * temp_fahrenheit + 
                  10.14333127 * relative_humidity - 
                  0.22475541 * temp_fahrenheit * relative_humidity - 
                  0.00683783 * temp_fahrenheit ** 2 - 
                  0.05481717 * relative_humidity ** 2 + 
                  0.00122874 * temp_fahrenheit ** 2 * relative_humidity + 
                  0.00085282 * temp_fahrenheit * relative_humidity ** 2 - 
                  0.00000199 * temp_fahrenheit ** 2 * relative_humidity ** 2)

            # Convert Heat Index from Fahrenheit to Celsius
            heat_index_celsius = (HI - 32) * 5/9

            return heat_index_celsius
        
        temp_weather_df['Heat Index'] = calculate_heat_index(temp_weather_df['MAX'], temp_weather_df['RH'])
        
        extreme_weather_df['Heat Index Caution'] = np.nan
        high_hum_df = temp_weather_df.loc[(temp_weather_df['Heat Index'] > 27) & 
                                               (temp_weather_df['Heat Index'] <= 32)]
        for date in high_hum_df['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Heat Index Caution'] = 1
                    
        extreme_weather_df['Heat Index Extreme Caution'] = np.nan
        high_hum_df = temp_weather_df.loc[(temp_weather_df['Heat Index'] > 32) & 
                                               (temp_weather_df['Heat Index'] <= 41)]
        for date in high_hum_df['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Heat Index Extreme Caution'] = 1
        
        
        extreme_weather_df['Heat Index Danger'] = np.nan
        high_hum_df = temp_weather_df.loc[(temp_weather_df['Heat Index'] > 41) & 
                                               (temp_weather_df['Heat Index'] <= 54)]
        for date in high_hum_df['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Heat Index Danger'] = 1
        
        
        extreme_weather_df['Heat Index Extreme Danger'] = np.nan
        high_hum_df = temp_weather_df.loc[(temp_weather_df['Heat Index'] > 54)]
        for date in high_hum_df['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Heat Index Extreme Danger'] = 1
        print("Heat Index done")

        # Wind Chill
        def calculate_wind_chill_index(temp_celsius, wind_speed_mps):
            """# Calculate wind chill index

            Args:
                temp_celsius (float): the temperature in celsius
                wind_speed_mps (float): the wind speed in mile per second

            Returns:
                float: American wind chill index
            """
            
            wind_chill_index_us = (
                35.74 + 
                0.6215 * temp_celsius - 
                35.75 * wind_speed_mps ** 0.16 + 
                0.4275 * temp_celsius * wind_speed_mps ** 0.16
            )
            return wind_chill_index_us
        
        temp_weather_df['Wind Chill'] = calculate_wind_chill_index(temp_weather_df['MIN'], temp_weather_df['MXSPD'])
        
        extreme_weather_df['Wind Chill Very Cold'] = np.nan
        high_hum_df = temp_weather_df.loc[(temp_weather_df['Heat Index'] > -35) & 
                                               (temp_weather_df['Heat Index'] <= -25)]
        for date in high_hum_df['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Chill Very Cold'] = 1
        
        
        extreme_weather_df['Wind Chill Frostbite Danger'] = np.nan
        high_hum_df = temp_weather_df.loc[(temp_weather_df['Heat Index'] > -60) & 
                                               (temp_weather_df['Heat Index'] <= -35)]
        for date in high_hum_df['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Chill Frostbite Danger'] = 1
                    
        
        extreme_weather_df['Wind Chill Great Frostbite Danger'] = np.nan
        high_hum_df = temp_weather_df.loc[(temp_weather_df['Heat Index'] <= -60)]
        for date in high_hum_df['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Chill Great Frostbite Danger'] = 1
        print("Wind Chill done")
        
        """
        # Storm
        extreme_weather_df['Tropical Storm'] = np.nan
        thunderstorm_39_54 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > (39 * 0.44704)) & 
                                               (temp_weather_df['MXSPD'] <= (54 * 0.44704))]
        for date in thunderstorm_39_54['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Tropical Storm'] = 1
        print("Tropical Storm done")
        
        extreme_weather_df['Severe Tropical Storm'] = np.nan
        thunderstorm_54_73 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > (54 * 0.44704)) & 
                                               (temp_weather_df['MXSPD'] <= (73 * 0.44704))]
        for date in thunderstorm_54_73['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Severe Tropical Storm'] = 1
        print("Severe Tropical Storm done")
        
        extreme_weather_df['Typhoon'] = np.nan
        thunderstorm_73_93 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > (73 * 0.44704)) & 
                                               (temp_weather_df['MXSPD'] <= (93 * 0.44704))]
        for date in thunderstorm_73_93['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Typhoon'] = 1
        print("Typhoon done")
        
        extreme_weather_df['Strong Typhoon'] = np.nan
        thunderstorm_93_114 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > (93 * 0.44704)) & 
                                               (temp_weather_df['MXSPD'] <= (114 * 0.44704))]
        for date in thunderstorm_93_114['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Strong Typhoon'] = 1
        print("Strong Typhoon done")
        
        extreme_weather_df['Super Typhoon'] = np.nan
        thunderstorm_114 = temp_weather_df.loc[temp_weather_df['MXSPD'] > (114 * 0.44704)]
        for date in thunderstorm_114['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Super Typhoon'] = 1
        print("Super Typhoon done")
        """
        
        # Wind Speed Level
        extreme_weather_df['Wind Level 0'] = np.nan
        thunderstorm_0 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > 0) & 
                                               (temp_weather_df['MXSPD'] <= 0.2)]
        for date in thunderstorm_0['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Level 0'] = 1
        print("Wind Level 0 done")
        
        extreme_weather_df['Wind Level 1'] = np.nan
        thunderstorm_1 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > 0.2) & 
                                               (temp_weather_df['MXSPD'] <= 1.5)]
        for date in thunderstorm_1['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Level 1'] = 1
        print("Wind Level 1 done")
        
        extreme_weather_df['Wind Level 2'] = np.nan
        thunderstorm_2 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > 1.5) & 
                                               (temp_weather_df['MXSPD'] <= 3.3)]
        for date in thunderstorm_2['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Level 2'] = 1
        print("Wind Level 2 done")
        
        extreme_weather_df['Wind Level 3'] = np.nan
        thunderstorm_3 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > 3.3) & 
                                               (temp_weather_df['MXSPD'] <= 5.4)]
        for date in thunderstorm_3['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Level 3'] = 1
        print("Wind Level 3 done")
        
        extreme_weather_df['Wind Level 4'] = np.nan
        thunderstorm_4 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > 5.4) & 
                                               (temp_weather_df['MXSPD'] <= 7.9)]
        for date in thunderstorm_4['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Level 4'] = 1
        print("Wind Level 4 done")
        
        extreme_weather_df['Wind Level 5'] = np.nan
        thunderstorm_5 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > 7.9) & 
                                               (temp_weather_df['MXSPD'] <= 10.7)]
        for date in thunderstorm_5['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Level 5'] = 1
        print("Wind Level 5 done")
        
        extreme_weather_df['Wind Level 6'] = np.nan
        thunderstorm_6 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > 10.7) & 
                                               (temp_weather_df['MXSPD'] <= 13.8)]
        for date in thunderstorm_6['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Level 6'] = 1
        print("Wind Level 6 done")
        
        extreme_weather_df['Wind Level 7'] = np.nan
        thunderstorm_7 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > 13.8) & 
                                               (temp_weather_df['MXSPD'] <= 17.1)]
        for date in thunderstorm_7['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Level 7'] = 1
        print("Wind Level 7 done")
        
        extreme_weather_df['Wind Level 8'] = np.nan
        thunderstorm_8 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > 17.1) & 
                                               (temp_weather_df['MXSPD'] <= 20.7)]
        for date in thunderstorm_8['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Level 8'] = 1
        print("Wind Level 8 done")
        
        extreme_weather_df['Wind Level 9'] = np.nan
        thunderstorm_9 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > 20.7) & 
                                               (temp_weather_df['MXSPD'] <= 24.4)]
        for date in thunderstorm_9['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Level 9'] = 1
        print("Wind Level 9 done")
        
        extreme_weather_df['Wind Level 10'] = np.nan
        thunderstorm_10 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > 24.4) & 
                                               (temp_weather_df['MXSPD'] <= 28.4)]
        for date in thunderstorm_10['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Level 10'] = 1
        print("Wind Level 10 done")
        
        extreme_weather_df['Wind Level 11'] = np.nan
        thunderstorm_11 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > 28.4) & 
                                               (temp_weather_df['MXSPD'] <= 32.6)]
        for date in thunderstorm_11['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Level 11'] = 1
        print("Wind Level 11 done")
        
        extreme_weather_df['Wind Level 12'] = np.nan
        thunderstorm_12 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > 32.6)]
        for date in thunderstorm_12['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Level 12'] = 1
        print("Wind Level 12 done")
        
        
        # Precipitation
        extreme_weather_df['Precipitation 50'] = np.nan
        prcp_0 = temp_weather_df.loc[(temp_weather_df['PRCP'] >= 0.05) & (temp_weather_df['PRCP'] < 0.1)]
        for date in prcp_0['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Precipitation 50'] = 1
        print("Precipitation 50 done")
        
        extreme_weather_df['Precipitation 100'] = np.nan
        prcp_1 = temp_weather_df.loc[(temp_weather_df['PRCP'] >= 0.1)]
        for date in prcp_1['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Precipitation 100'] = 1
        print("Precipitation 100 done")
        
        extreme_weather_df["STATION_ID"] = element
    
    return None


if __name__ == "__main__":
    time_index = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")
    temp_time_index = pd.DataFrame()
    temp_time_index["DATETIME"] = time_index
    
    # 1. Weather data and weather meta data
    # 1.1 Obtain the NCDC data
    weather_meta_df = NCDC_weather_meta_data_obtain("./data/isd-history.csv", 2022, 2023)
    weather_meta_gdf = df_to_gdf(weather_meta_df, "LON", "LAT")
    # 1.2 Filter by region
    provincial_shp = gpd.read_file("./data/guangxi_administration/guangxi.shp")
    provincial_weather_meta_gdf = weather_meta_gdf[weather_meta_gdf.geometry.within(provincial_shp.unary_union)]
    # 1.3 Obtain weather data from internet
    #weather_df = NCDC_weather_data_obtain(provincial_weather_meta_gdf, 2022, 2023)

    
    
    
    # 2. Transformer data
    transformer_meta_df = pd.read_excel("./data/transformer_meta.xlsx")
    
    
    transformer_df = pd.read_excel("./data/transformer_raw.xlsx")
    print(transformer_df)
    
    # 3. Extreme weather data from internet
    extreme_weather_internet_df = pd.read_excel("./data/extreme_weather_internet.xlsx")
    
    # 4. Calculated extreme weather
    extreme_weather_detect(meta_df, 2022, 2023)
    
    # 5. Holiday data
    holiday_df = pd.read_excel("./holiday.xlsx")
