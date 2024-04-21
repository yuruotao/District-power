import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Import modules
import script.analysis as analysis

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
    start_time = '2022-01-01 00:00:00'
    imputed_df = pd.read_excel("./result/imputation/imputed_data_Forward-Backward.xlsx")
    
    district_df = district_aggregate(imputed_df, 2, "./result/aggregate/")
    city_df = district_aggregate(imputed_df, 1,"./result/aggregate/")
    
    select_df = district_df[["Datetime", "0-0", "1-0", "2-0", "3-0", "4-0", "5-0", "6-0", "7-0", "8-0", "9-0"]]
    select_df = select_df.rename(columns={
        "0-0":"0", "1-0":"1", "2-0":"2", 
        "3-0":"3", "4-0":"4", "5-0":"5", 
        "6-0":"6", "7-0":"7", "8-0":"8", 
        "9-0":"9"
    })
    
    # Load profile for all districts
    #analysis.average_load_profiles(select_df, "./result/select/")
    
    # Load profile in different scales

    select_df = select_df[["Datetime", "0", "2", "3", "5", "9"]]

    analysis.specific_load_profile_plot(select_df, 
                                        '2022-07-04 00:00:00', '2022-07-04 23:00:00', 
                                        '2022-07-10 00:00:00', '2022-07-10 23:00:00', 
                                        "Day", "./result/select/")
    
    analysis.specific_load_profile_plot(select_df, 
                                        '2022-07-04 00:00:00', '2022-07-10 23:00:00', 
                                        '2022-07-11 00:00:00', '2022-07-17 23:00:00', 
                                        "Week", "./result/select/")

    analysis.specific_load_profile_plot(select_df, 
                                        '2022-07-01 00:00:00', '2022-07-31 23:00:00', 
                                        '2022-08-01 00:00:00', '2022-08-31 23:00:00', 
                                        "Month", "./result/select/")

    