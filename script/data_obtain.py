import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import gdelt
import requests

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



if __name__ == "__main__":
    #NCDC_weather_data_obtain("./data/isd-history.csv", "./result/NCDC_weather_data/", 2022, 2023+1)
    NCDC_weather_data_station_merge("./data/isd-history.csv",
                                    "./result/NCDC_weather_data/", 
                                    "./result/NCDC_weather_data/stations/",
                                    2022, 2023+1)