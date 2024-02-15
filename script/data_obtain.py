import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import gdelt
import requests

def NCDC_weather_data_obtain(meta_path, output_path, start_year, stop_year):
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
    NCDC_weather_data_obtain("./data/isd-history.csv", "./result/weather_data/", 2022, 2023+1)