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
from io import StringIO
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from os import listdir
from os.path import isfile, join

sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
sns.set_theme(style="white")
mpl.rcParams['font.family'] = 'Times New Roman'

Base = declarative_base()

class weather_meta(Base):
    __tablename__ = 'weather_meta'
    STATION_ID = Column(String, primary_key=True, unique=True, nullable=False)
    USAF = Column(String)
    WBAN = Column(String)
    STATION_NAME = Column(String)
    CTRY = Column(String)
    STATE = Column(String)
    ICAO = Column(String)
    LAT = Column(Float)
    LON = Column(Float)
    ELEV = Column(Float)
    BEGIN = Column(Integer)
    END = Column(Integer)

# Datetime TEMP DEWP SLP STP VISIB WDSP MXSPD GUST MAX MIN PRCP SNDP RH
class weather(Base):
    __tablename__ = 'weather'
    ID = Column(Integer, primary_key=True, unique=True, nullable=False)
    STATION_ID = Column(String)
    DATETIME = Column(DateTime)
    TEMP = Column(Float)
    DEWP = Column(Float)
    SLP = Column(Float)
    STP = Column(Float)
    VISIB = Column(Float)
    WDSP = Column(Float)
    MXSPD = Column(Float)
    GUST = Column(Float)
    MAX = Column(Float)
    MIN = Column(Float)
    PRCP = Column(Float)
    SNDP = Column(Float)
    RH = Column(Float)

class transformer_meta(Base):
    __tablename__ = 'transformer_meta'
    TRANSFORMER_ID = Column(String, primary_key=True, unique=True, nullable=False)
    CITY = Column(Integer)
    DISTRICT = Column(Integer)
    TRANSFORMER = Column(Integer)
    YXRL = Column(Integer)
    CLOSEST_STATION = Column(String)
    DELETE = Column(Integer)

class transformer_raw(Base):
    __tablename__ = 'transformer_raw'
    ID = Column(String, primary_key=True, unique=True, nullable=False)
    TRANSFORMER_ID = Column(String)
    DATETIME = Column(DateTime)
    LOAD = Column(Float)

class extreme_weather_internet(Base):
    __tablename__ = 'extreme_weather_internet'
    IDENTIFIER = Column(String, primary_key=True, unique=True, nullable=False)
    HAZARD = Column(String)
    DATETIME = Column(DateTime)

class extreme_weather_calculated(Base):
    __tablename__ = 'extreme_weather_calculated'
    IDENTIFIER = Column(String, primary_key=True, unique=True, nullable=False)
    DATETIME = Column(DateTime)
    STATION_ID = Column(String)
    High_Temperature = Column(Integer)
    Low_Temperature = Column(Integer)
    High_Humidity = Column(Integer)
    Heat_Index_Caution = Column(Integer)
    Heat_Index_Extreme_Caution = Column(Integer)
    Heat_Index_Danger = Column(Integer)
    Heat_Index_Extreme_Danger = Column(Integer)
    Wind_Chill_Very_Cold = Column(Integer)
    Wind_Chill_Frostbite_Danger = Column(Integer)
    Wind_Chill_Great_Frostbite_Danger = Column(Integer)
    Wind_Level_0 = Column(Integer)
    Wind_Level_1 = Column(Integer)
    Wind_Level_2 = Column(Integer)
    Wind_Level_3 = Column(Integer)
    Wind_Level_4 = Column(Integer)
    Wind_Level_5 = Column(Integer)
    Wind_Level_6 = Column(Integer)
    Wind_Level_7 = Column(Integer)	
    Wind_Level_8 = Column(Integer)	
    Wind_Level_9 = Column(Integer)
    Wind_Level_10 = Column(Integer)	
    Wind_Level_11 = Column(Integer)
    Wind_Level_12 = Column(Integer)
    Precipitation_50 = Column(Integer)	
    Precipitation_100 = Column(Integer)

class holiday(Base):
    __tablename__ = 'holiday'
    ID = Column(Integer, primary_key=True, unique=True, nullable=False)
    HOLIDAY = Column(String)
    DATETIME = Column(DateTime)
##################################################################################################
# 1. Weather data
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
    """Obtain the weather data of NCDC

    Args:
        meta_df (dataframe): contain the meta data for web info
        start_year (int): the start year for weather data
        stop_year (int): stop year for weather data

    Returns:
        dataframe: contain the weather data
    """
    time_index = pd.date_range(start=str(start_year) + "-01-01", end=str(stop_year) + "-12-31", freq="D")
    temp_time_index = pd.DataFrame()
    temp_time_index["Datetime"] = time_index
    
    base = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/"
    
    for index, row in meta_df.iterrows():
        Station_ID = str(row["Station_ID"])
        print(Station_ID)
        usaf_value = row['USAF']
        wban_value = row['WBAN']
        
        for year in range(start_year, stop_year+1):
            # Obtain data from website
            base_year = base + str(year) + "/"
            url = base_year + str(usaf_value) + str(wban_value) + ".csv"
            response = requests.get(url)
            
            # Convert to dataframe for concatenate
            csv_data = response.text
            csv_data_io = StringIO(csv_data)
            temp_df = pd.read_csv(csv_data_io)
            
            try: 
                temp_df = temp_df[["DATE",
                            "TEMP", "DEWP", "SLP", "STP", "VISIB", 
                            "WDSP", "MXSPD", "GUST", "MAX", "MIN", "PRCP", 
                            "SNDP"]]
            except:
                temp_list = ["TEMP", "DEWP", "SLP", "STP", "VISIB", 
                            "WDSP", "MXSPD", "GUST", "MAX", "MIN", "PRCP", 
                            "SNDP"]
                for col in temp_list:
                    temp_df[col] = np.nan
            
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
            
            temp_df["STATION_ID"] = Station_ID
            
            if index == 0 and year == start_year:
                weather_df = temp_df
            else:
                weather_df = pd.concat([weather_df, temp_df], axis=0)

    weather_df = weather_df.reset_index(drop=True)
        
    return weather_df
##################################################################################################
# 2. Transformer data
##################################################################################################
# 3. Extreme weather from the internet
##################################################################################################
# 4. Calculated extreme weather
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
        
def extreme_weather_detect(weather_meta_df, weather_df, start_year, end_year):
    """Detect the extreme weather based on conditions

    Args:
        weather_meta_df (dataframe): contain the meta data for weather stations
        weather_df (dataframe): contain the weather data
        start_year (string): the start date of extreme weather
        end_year (string): the end date of extreme weather

    Returns:
        dataframe
    """
    
    time_index = pd.date_range(start=str(start_year) + "-01-01", end=str(end_year) + "-12-31", freq='d')
    # Create a DataFrame with the time series column
    datetime_df = pd.DataFrame({'DATETIME': time_index})
    station_set = set(weather_meta_df["STATION_ID"])
    
    iter = 0
    for element in station_set:
        temp_weather_df = weather_df[weather_df["STATION_ID"]==element]
        temp_weather_df = temp_weather_df.astype({"DATETIME":"datetime64[ms]"})
        temp_weather_df = pd.merge(datetime_df, temp_weather_df, on="DATETIME", how="left")
        extreme_weather_df = datetime_df
        
        extreme_weather_df['High_Temperature'] = np.nan
        extreme_weather_df['Low_Temperature'] = np.nan
        extreme_weather_df['High_Humidity'] = np.nan
        extreme_weather_df['Heat_Index_Caution'] = np.nan
        extreme_weather_df['Heat_Index_Extreme_Caution'] = np.nan
        extreme_weather_df['Heat_Index_Danger'] = np.nan
        extreme_weather_df['Heat_Index_Extreme_Danger'] = np.nan
        extreme_weather_df['Wind_Chill_Very_Cold'] = np.nan
        extreme_weather_df['Wind_Chill_Frostbite_Danger'] = np.nan
        extreme_weather_df['Wind_Chill_Great_Frostbite_Danger'] = np.nan
        extreme_weather_df['Wind_Level_0'] = np.nan
        extreme_weather_df['Wind_Level_1'] = np.nan
        extreme_weather_df['Wind_Level_2'] = np.nan
        extreme_weather_df['Wind_Level_3'] = np.nan
        extreme_weather_df['Wind_Level_4'] = np.nan
        extreme_weather_df['Wind_Level_5'] = np.nan
        extreme_weather_df['Wind_Level_6'] = np.nan
        extreme_weather_df['Wind_Level_7'] = np.nan
        extreme_weather_df['Wind_Level_8'] = np.nan
        extreme_weather_df['Wind_Level_9'] = np.nan
        extreme_weather_df['Wind_Level_10'] = np.nan
        extreme_weather_df['Wind_Level_11'] = np.nan
        extreme_weather_df['Wind_Level_12'] = np.nan
        extreme_weather_df['Precipitation_50'] = np.nan
        extreme_weather_df["Precipitation_100"] = np.nan

        # High Temperature
        MAX_percentile_95 = temp_weather_df['MAX'].quantile(0.95)
        high_temp_dates = temp_weather_df[temp_weather_df['MAX'] > MAX_percentile_95]['DATETIME']
        extreme_weather_df.loc[extreme_weather_df['DATETIME'].dt.date.isin(high_temp_dates.dt.date), 'High_Temperature'] = 1

        # Low Temperature
        MIN_percentile_5 = temp_weather_df['MIN'].quantile(0.05)
        low_temp_dates = temp_weather_df[temp_weather_df['MIN'] < MIN_percentile_5]['DATETIME']
        extreme_weather_df.loc[extreme_weather_df['DATETIME'].dt.date.isin(low_temp_dates.dt.date), 'Low_Temperature'] = 1

        # High Humidity
        RH_percentile_95 = temp_weather_df['RH'].quantile(0.95)
        high_humidity_dates = temp_weather_df[temp_weather_df['RH'] > RH_percentile_95]['DATETIME']
        extreme_weather_df.loc[extreme_weather_df['DATETIME'].dt.date.isin(high_humidity_dates.dt.date), 'High_Humidity'] = 1

        # Heat Index
        temp_weather_df['Heat_Index'] = calculate_heat_index(temp_weather_df['MAX'], temp_weather_df['RH'])

        def label_heat_index(row, dates, column):
            extreme_weather_df.loc[extreme_weather_df['DATETIME'].dt.date.isin(dates.dt.date), column] = 1

        heat_index_ranges = {
            'Heat_Index_Caution': (27, 32),
            'Heat_Index_Extreme_Caution': (32, 41),
            'Heat_Index_Danger': (41, 54),
            'Heat_Index_Extreme_Danger': (54, float('inf'))
        }

        for label, (low, high) in heat_index_ranges.items():
            dates = temp_weather_df[(temp_weather_df['Heat_Index'] > low) & (temp_weather_df['Heat_Index'] <= high)]['DATETIME']
            label_heat_index(extreme_weather_df, dates, label)

        # Wind Chill
        temp_weather_df['Wind_Chill'] = calculate_wind_chill_index(temp_weather_df['MIN'], temp_weather_df['MXSPD'])

        wind_chill_ranges = {
            'Wind_Chill_Very_Cold': (-35, -25),
            'Wind_Chill_Frostbite_Danger': (-60, -35),
            'Wind_Chill_Great_Frostbite_Danger': (float('-inf'), -60)
        }

        for label, (low, high) in wind_chill_ranges.items():
            dates = temp_weather_df[(temp_weather_df['Heat_Index'] > low) & (temp_weather_df['Heat_Index'] <= high)]['DATETIME']
            label_heat_index(extreme_weather_df, dates, label)

        # Wind Speed Level
        wind_speed_ranges = {
            'Wind_Level_0': (0, 0.2),
            'Wind_Level_1': (0.2, 1.5),
            'Wind_Level_2': (1.5, 3.3),
            'Wind_Level_3': (3.3, 5.4),
            'Wind_Level_4': (5.4, 7.9),
            'Wind_Level_5': (7.9, 10.7),
            'Wind_Level_6': (10.7, 13.8),
            'Wind_Level_7': (13.8, 17.1),
            'Wind_Level_8': (17.1, 20.7),
            'Wind_Level_9': (20.7, 24.4),
            'Wind_Level_10': (24.4, 28.4),
            'Wind_Level_11': (28.4, 32.6),
            'Wind_Level_12': (32.6, float('inf')),
            
        }

        for label, (low, high) in wind_speed_ranges.items():
            dates = temp_weather_df[((temp_weather_df['MXSPD'] > low) & (temp_weather_df['MXSPD'] <= high))]['DATETIME']
            label_heat_index(extreme_weather_df, dates, label)
            
        # Precipitation
        precipitation_ranges ={
            "Precipitation_50": (0.05, 0.1),
            "Precipitation_100": (0.1, float('inf'))
        }
        
        for label, (low, high) in precipitation_ranges.items():
            dates = temp_weather_df[(temp_weather_df['PRCP'] > low) & (temp_weather_df['PRCP'] <= high)]['DATETIME']
            print(label, low, high)
            print(dates)
            label_heat_index(extreme_weather_df, dates, label)
        
        extreme_weather_df["STATION_ID"] = element
        
        if iter == 0:
            final_df = extreme_weather_df
            iter += 1
        else:
            final_df = pd.concat([final_df, extreme_weather_df], axis=0, ignore_index=True)
        print(final_df)
    return final_df
##################################################################################################
# 5. Holidays
##################################################################################################

if __name__ == "__main__":
    time_index = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")
    time_df = pd.DataFrame()
    time_df["Datetime"] = time_index
    
    engine = create_engine('sqlite:///./data/Transformer_DB/Transformer_DB.db')
    Base.metadata.create_all(engine)
    
    # Create a new session
    Session = sessionmaker(bind=engine)
    session = Session()
    ##################################################################################################
    # 1. Weather data and weather meta data
    # 1.1 Obtain the NCDC data
    weather_meta_df = NCDC_weather_meta_data_obtain("./data/isd-history.csv", 2022, 2023)
    weather_meta_gdf = df_to_gdf(weather_meta_df, "LON", "LAT")
    # 1.2 Filter by region
    provincial_shp = gpd.read_file("./data/guangxi_administration/guangxi.shp")
    provincial_weather_meta_gdf = weather_meta_gdf[weather_meta_gdf.geometry.within(provincial_shp.unary_union)]
    provincial_weather_meta_gdf = provincial_weather_meta_gdf.reset_index(drop=True)
    provincial_weather_meta_df = provincial_weather_meta_gdf.drop(["geometry"], axis=1)
    provincial_weather_meta_df = provincial_weather_meta_df.reset_index(drop=True)
    provincial_weather_meta_df.columns = provincial_weather_meta_df.columns.str.upper()
    # 1.3 Obtain weather data from internet
    weather_df = NCDC_weather_data_obtain(provincial_weather_meta_gdf, 2022, 2023)
    weather_df = weather_df.rename({"DATE": "DATETIME"}, axis=1)
    weather_df.columns = weather_df.columns.str.upper()
    # 1.4 Store them in a database
    provincial_weather_meta_df.to_sql('weather_meta', con=engine, if_exists='replace', index=False)
    weather_df.to_sql('weather', con=engine, if_exists='replace', index=False)
    ##################################################################################################
    # 2. Transformer data
    # 2.1 Load the transformer meta data and transformer data
    transformer_meta_df = pd.read_excel("./data/transformer_meta.xlsx")
    transformer_meta_df["TRANSFORMER_ID"] = transformer_meta_df.apply(lambda row: str(int(row["City"])) + '-' + str(int(row["District"]))
                                                                      + "-" + str(int(row["Transformer"])), axis=1)
    transformer_meta_df.columns = transformer_meta_df.columns.str.upper()
    transformer_df = pd.read_excel("./data/transformer_raw.xlsx")
    dfs = []
    for column in transformer_df.columns[transformer_df.columns != 'Datetime']:
        # Create a new DataFrame with Transformer_ID, LOAD, and the actual data
        temp_df = pd.DataFrame({
            'Transformer_ID': [column] * len(transformer_df),
            'LOAD': transformer_df[column],
            'Datetime': transformer_df['Datetime']  # Include Datetime if needed
        })
        # Append this DataFrame to the list
        dfs.append(temp_df[['Transformer_ID', 'LOAD', 'Datetime']])
    # Concatenate all the individual DataFrames into a final DataFrame
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.columns = final_df.columns.str.upper()
    # 2.2 Store them in a database
    transformer_meta_df.to_sql('transformer_meta', con=engine, if_exists='replace', index=False)
    final_df.to_sql('transformer_raw', con=engine, if_exists='replace', index=False)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
    ##################################################################################################
    # 3. Extreme weather data from internet
    # 3.1 Import the data and save to database
    extreme_weather_internet_df = pd.read_excel("./data/extreme_weather_internet.xlsx")
    extreme_weather_internet_df = pd.merge(time_df, extreme_weather_internet_df, on="Datetime", how="left")
    extreme_weather_internet_df.columns = extreme_weather_internet_df.columns.str.upper()
    extreme_weather_internet_df = extreme_weather_internet_df.rename({"EVENT": "HAZARD"}, axis=1)
    extreme_weather_internet_df.to_sql('extreme_weather_internet', con=engine, if_exists='replace', index=False)
    ##################################################################################################
    # 4. Calculated extreme weather
    provincial_weather_meta_df_query = "SELECT * FROM weather_meta"
    provincial_weather_meta_df = pd.read_sql(provincial_weather_meta_df_query, engine)
    weather_df_query = "SELECT * FROM weather"
    weather_df = pd.read_sql(weather_df_query, engine)
    extreme_weather_detect_df = extreme_weather_detect(provincial_weather_meta_df, weather_df, 2022, 2023)
    extreme_weather_detect_df.columns = extreme_weather_detect_df.columns.str.upper()
    extreme_weather_detect_df.to_sql('extreme_weather_calculated', con=engine, if_exists='replace', index=False)
    ##################################################################################################
    # 5. Holiday data
    # 5.1 Import the data and save to database
    holiday_df = pd.read_excel("./data/holiday.xlsx")
    holiday_df = pd.merge(time_df, holiday_df, on="Datetime", how="left")
    holiday_df.columns = holiday_df.columns.str.upper()
    holiday_df.to_sql('holiday', con=engine, if_exists='replace', index=False)
    ##################################################################################################