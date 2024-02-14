import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import gdelt
import datetime

def NCDC_weather_data_obtain(meta_path, download_path, start_year, stop_year):
    # meta_df is obtained from
    # https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/
    meta_df = pd.read_csv(meta_path)
    meta_df = meta_df[["CTRY"]== "CH"]
    meta_df.reset_index()
    
    # download the data with respect to "USAF"
    
    

    "LAT	LON	ELEV(M)	BEGIN	END"
    
    return None
    
def GDELT_data_obtain():


    # Set up the GDELT API
    gd = gdelt.gdelt()

    # Define the date range
    start_date = "20220101"
    end_date = "20231231"

    # Define the location (Guangxi China)
    location = "Guangxi China"

    # Search for events during the specified time period
    results = gd.Search(date=(start_date, end_date), table='events', coverage=True)

    # Convert the results to a DataFrame
    df = pd.DataFrame(results)

    # Filter the DataFrame for events related to Guangxi China
    guangxi_df = df[df['Actor2Geo_FullName'].str.contains(location, na=False) | df['Actor1Geo_FullName'].str.contains(location, na=False)]

    # Display the filtered DataFrame
    print(guangxi_df.head())
    
    return None

if __name__ == "__main__":
    
    GDELT_data_obtain()