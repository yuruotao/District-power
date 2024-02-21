import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

# Import modules
import script.analysis as analysis
import script.imputation as imputation
import script.missing_value as missing_value
import script.profiling as profiling
import script.basic_statistics as basic_statistics
import script.resample as resample
import script.uniform as uniform

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
    
    imputed_df = pd.read_excel("./result/imputation/imputed_data_Forward-Backward.xlsx")
    
    district_df = district_aggregate(imputed_df, 2, "./result/aggregate/")
    city_df = district_aggregate(imputed_df, 1,"./result/aggregate/")
    
    select_df = district_df[district_df["0-0", "1-0", "2-0", "3-0", "4-0", "5-0", "6-0", "7-0", "8-0", "9-0"]]
    
    