import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns


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

def separate_by_city(input_df, meta_df, output_path):
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


# Separate data into districts
def raw_data(district_df):
    time_index = pd.date_range(start='2022-01-01 00:00', end='2022-12-31 23:00', freq='H')
    # Create a DataFrame with the time series column
    time_series_df = pd.DataFrame({'Datetime': time_index})

    district_set = set(district_df['district_name'])
    district_list = list(district_set)
    
    for district in district_list:
        temp_dir = "./result/district/raw_by_district/" + district

        district_df = district_df.loc[district_df['district_name'] == district]
        print(district_df)
        TQMC_set = set(district_df['TQMC'])
        TQMC_list = list(TQMC_set)
        for TQMC in TQMC_list:
            temp_dir_TQMC = temp_dir + "/" + TQMC
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            TQMC_df = district_df.loc[district_df['TQMC'] == TQMC]
            TQMC_df.sort_values(by='Datetime', inplace = True)
            TQMC_df = TQMC_df.reset_index(drop=True)
            TQMC_df.to_excel(temp_dir_TQMC + ".xlsx", index=False)
            print(TQMC)
            print(TQMC_df)

# Time period selection and data censoring
def raw_data_censor(district_df):
    time_index = pd.date_range(start='2022-01-01 00:00', end='2022-12-31 23:00', freq='H')
    # Create a DataFrame with the time series column
    time_series_df = pd.DataFrame({'Datetime': time_index})
    
    district_df['Year'] = district_df['Datetime'].dt.year
    district_df['Month'] = district_df['Datetime'].dt.month
    district_df['Day'] = district_df['Datetime'].dt.day
    district_df['Hour'] = district_df['Datetime'].dt.hour
    district_set = set(district_df['district_name'])
    district_list = list(district_set)

    # Censor by year
    district_df = district_df.loc[district_df['Year'] >= 2022]
    district_df = district_df.loc[district_df['Year'] < 2023]
    meta_df = pd.DataFrame(columns=['District','Transformer','WQ','GDDWBM','YHBH','TQMC','YHMC','YHLB','YXRL', "Block_name"])
    
    district_num = 0

    for district in district_list:
        temp_dir = "./result/district/raw_by_district_censored/District-" + str(district_num)
        TQMC_num = 0
        
        district_df = district_df.loc[district_df['district_name'] == district]
        print(district_df)
        TQMC_set = set(district_df['TQMC'])
        TQMC_list = list(TQMC_set)
        for TQMC in TQMC_list:
            temp_dir_TQMC = temp_dir + "/"
            
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            TQMC_df = district_df.loc[district_df['TQMC'] == TQMC]
            TQMC_df.sort_values(by='Datetime', inplace = True)
            
            # Meta
            meta_list = [district_num,
             TQMC_num,
             TQMC_df.iloc[0]['WQ'],
             TQMC_df.iloc[0]['GDDWBM'],
             TQMC_df.iloc[0]['YHBH'],
             TQMC_df.iloc[0]['TQMC'],
             TQMC_df.iloc[0]['YHMC'],
             TQMC_df.iloc[0]['YHLB'],
             TQMC_df.iloc[0]['YXRL'], 
             TQMC_df.iloc[0]["district_name"]]
            meta_data = pd.DataFrame(meta_list).T
            meta_data.columns = ['District','Transformer','WQ','GDDWBM','YHBH','TQMC','YHMC','YHLB','YXRL', "Block_name"]
            meta_df = pd.concat([meta_df, meta_data], ignore_index=True)
            print(meta_df)
            
            TQMC_df = TQMC_df.reset_index(drop=True)
            TQMC_df = TQMC_df.drop_duplicates(subset="Datetime")
            TQMC_df = pd.merge(time_series_df, TQMC_df, on='Datetime', how="left")
            TQMC_df['Year'] = TQMC_df['Datetime'].dt.year
            TQMC_df['Month'] = TQMC_df['Datetime'].dt.month
            TQMC_df['Day'] = TQMC_df['Datetime'].dt.day
            TQMC_df['Hour'] = TQMC_df['Datetime'].dt.hour
            TQMC_df['YXRL'] = TQMC_df.iloc[0]['YXRL']
            TQMC_df = TQMC_df[["Year", "Month", "Day", "Hour", "Power", "YXRL"]]
            
            TQMC_df = TQMC_df.reset_index(drop=True)
            TQMC_df.to_excel(temp_dir_TQMC +"Transformer-"+ str(district_num) + "-" + str(TQMC_num) +  ".xlsx", index=False)
            print(TQMC)
            print(TQMC_df)
            TQMC_num = TQMC_num + 1
        district_num = district_num + 1
    meta_df.to_excel("./result/district/meta.xlsx",index=False)
    
# Impute data by district  
def raw_data_file_create(district_df, meta_df):
    time_index = pd.date_range(start='2022-01-01 00:00', end='2022-12-31 23:00', freq='H')
    # Create a DataFrame with the time series column
    power_2022_district_df = pd.DataFrame({'Datetime': time_index})
    time_series_df = power_2022_district_df
    
    district_df['Year'] = district_df['Datetime'].dt.year
    district_set = set(district_df['district_name'])
    district_list = list(district_set)
    
    # Censor by year
    district_df = district_df.loc[district_df['Year'] >= 2022]
    district_df = district_df.loc[district_df['Year'] < 2023]
    
    for district in district_list:
        temp_dir = "./result/district/raw_imputed/" + district
        profile_dir = "./result/district/profiling/"
        power_2022_TQMC_df = time_series_df
        temp_district_df = district_df.loc[district_df['district_name'] == district]
        TQMC_set = set(temp_district_df['TQMC'])
        TQMC_list = list(TQMC_set)
        
        for TQMC in TQMC_list:
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            TQMC_df = temp_district_df.loc[temp_district_df['TQMC'] == TQMC]
            TQMC_df = TQMC_df[["Datetime", "Power"]]
            TQMC_df.sort_values(by='Datetime', inplace = True)
            
            # Locate the numbering
            TQMC_meta = meta_df.loc[meta_df['TQMC'] == TQMC]
            district_num = TQMC_meta["District"].item()
            transformer_num = TQMC_meta["Transformer"].item()
            TQMC_df = TQMC_df.rename(columns={"Power":str(district_num) + "-" + str(transformer_num)})
            TQMC_df = TQMC_df.reset_index(drop=True)
            TQMC_df = TQMC_df.drop_duplicates(subset="Datetime")
            TQMC_df = pd.merge(time_series_df, TQMC_df, on='Datetime', how="right")
            
            power_2022_TQMC_df = pd.merge(power_2022_TQMC_df, TQMC_df, on='Datetime', how="left")
            power_2022_district_df = pd.merge(power_2022_district_df, TQMC_df, on='Datetime', how="left")
        
        # Single district
        power_2022_TQMC_df = power_2022_TQMC_df.drop_duplicates(subset="Datetime")
        power_2022_TQMC_df = power_2022_TQMC_df.reset_index(drop=True)
        
        # Rearrange order
        cols = power_2022_TQMC_df.columns.tolist()
        arranged_cols = ["Datetime"]
        for i in range(0, len(cols)-1):
            arranged_cols.append(str(district_num)+"-"+str(i))
        print(arranged_cols)
        print(power_2022_TQMC_df)
        power_2022_TQMC_df = power_2022_TQMC_df[arranged_cols]

        # Profiling
        if not os.path.exists(profile_dir):
            os.makedirs(profile_dir)
            
        #profile = ProfileReport(
        #        power_2022_TQMC_df,
        #        tsmode=True,
        #        sortby="Datetime",
        #        title="Time-Series EDA for District " + str(district_num),
        #    )
        #profile.to_file(profile_dir + "District-" + str(district_num) + ".html")
        
        power_2022_TQMC_df = power_2022_TQMC_df.drop(['Datetime'], axis=1)
        print(power_2022_TQMC_df)
        
        
        
        print(str(district_num) + " Done")


    
    print(power_2022_district_df)
    # All districts
    power_2022_district_df = power_2022_district_df.drop_duplicates(subset="Datetime")
    power_2022_district_df = power_2022_district_df.reset_index(drop=True)
    datetime_column = power_2022_district_df["Datetime"]
    power_2022_district_df = power_2022_district_df.drop(columns=["Datetime"])
    sorted_columns = sorted(power_2022_district_df.columns, key=lambda x: [int(i) for i in x.split("-")])
    power_2022_district_df = power_2022_district_df[sorted_columns]
    power_2022_district_df = pd.concat([datetime_column, power_2022_district_df], axis=1)
    power_2022_district_df.to_excel("./result/district/raw_data_1.xlsx", index=False)
    
    #msno.matrix(power_2022_district_df)
    #plt.show()
    #msno.heatmap(power_2022_district_df)
    #plt.show()
 


if __name__ == "__main__":
    # Import raw data and meta data
    raw_data_path = "./data/raw_data.xlsx"
    meta_path = "./data/meta.xlsx"
    
    #raw_data_df = pd.read_excel(raw_data_path)
    #meta_df = pd.read_excel(meta_path)
    
    # Define the time interval
    start_time = '2022-01-01 00:00:00'
    end_time = '2023-11-10 08:00:00'

    # Create a new DataFrame based on the time interval
    #raw_data_df = raw_data_df.loc[(raw_data_df['Datetime'] >= start_time) & (raw_data_df['Datetime'] <= end_time)]
    #raw_data_df = raw_data_df.reset_index()
    #missing_value.missing_value_visualization(raw_data_df, "./result/missing_value")
    #profiling.profiling(raw_data_df, "./result/profile/raw", "json")
    #profiling.profiling(raw_data_df, "./result/profile/raw", "html")
    #basic_statistics.basic_statistics(raw_data_df, "./result/basic_statistics/raw")
    
    
    # Delete the columns whose missing value takes up more than 20%
    raw_data_adjusted_df = pd.read_excel("./data/raw_data_adjusted.xlsx")
    raw_data_adjusted_df = raw_data_adjusted_df.loc[(raw_data_adjusted_df['Datetime'] >= start_time) & (raw_data_adjusted_df['Datetime'] <= end_time)]
    raw_data_adjusted_df = raw_data_adjusted_df.reset_index()
    #basic_statistics.basic_statistics(raw_data_df, "./result/basic_statistics/adjusted")
    
    #imputed_df = imputation.imputation(raw_data_adjusted_df, save_path="./result/imputation", imputation_method="Forward")
    #imputed_df = pd.read_excel("./result/imputation/imputed_data_Forward.xlsx")
    #basic_statistics.basic_statistics(imputed_df, "./result/basic_statistics/imputation/Forward")
    
    #imputed_df = imputation.imputation(raw_data_adjusted_df, save_path="./result/imputation", imputation_method="Backward")
    #imputed_df = pd.read_excel("./result/imputation/imputed_data_Backward.xlsx")
    #basic_statistics.basic_statistics(imputed_df, "./result/basic_statistics/imputation/Backward")

    #imputed_df = imputation.imputation(raw_data_adjusted_df, save_path="./result/imputation", imputation_method="Forward-Backward")
    #imputed_df = pd.read_excel("./result/imputation/imputed_data_Forward-Backward.xlsx")
    #basic_statistics.basic_statistics(imputed_df, "./result/basic_statistics/imputation/Forward-Backward")
    
    #imputed_df = imputation.imputation(raw_data_adjusted_df, save_path="./result/imputation", imputation_method="Average")
    #imputed_df = pd.read_excel("./result/imputation/imputed_data_Average.xlsx")
    #basic_statistics.basic_statistics(imputed_df, "./result/basic_statistics/imputation/Average")
    
    #imputed_df = imputation.imputation(raw_data_adjusted_df, save_path="./result/imputation", imputation_method="MICE")
    #imputed_df = pd.read_excel("./result/imputation/imputed_data_MICE.xlsx")
    #basic_statistics.basic_statistics(imputed_df, "./result/basic_statistics/imputation/MICE")
    
    #imputed_df = imputation.imputation(raw_data_adjusted_df, save_path="./result/imputation", imputation_method="BiScaler")
    #imputed_df = pd.read_excel("./result/imputation/imputed_data_BiScaler.xlsx")
    #basic_statistics.basic_statistics(imputed_df, "./result/basic_statistics/imputation/BiScaler")
    
    #imputed_df = imputation.imputation(raw_data_adjusted_df, save_path="./result/imputation", imputation_method="FEDOT")
    #imputed_df = pd.read_excel("./result/imputation/imputed_data_FEDOT.xlsx")
    #basic_statistics.basic_statistics(imputed_df, "./result/basic_statistics/imputation/FEDOT")
    
    resample_df_list = resample.resample(raw_data_adjusted_df, output_path="./result/resample", freq_list=['D','W'])
    

    