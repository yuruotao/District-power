import pandas as pd
import numpy as np
import os


def uniform(meta_df, city_df, output_path):
    """Uniform the data by dividing the rated capacity

    Args:
        meta_df (dataframe): meta data containing rated capacity data
        city_df (dataframe): contain the city-level aggregated data
        output_path (string): path to save the output

    Returns:
        dataframe: contain the uniformed data 
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    uniform_df = city_df
    
    meta_df = meta_df[meta_df['Delete'].isna()]
    city_set = set(meta_df["City"])
    for city in city_set:
        print(city)
        temp_meta_df = meta_df[meta_df['City'] == city]
        temp_meta_df = temp_meta_df.astype({"YXRL": float})
        city_rated_capacity = temp_meta_df["YXRL"].sum()
        print(city_rated_capacity)
        uniform_df[str(city)] = uniform_df[str(city)].div(city_rated_capacity)
    
    uniform_df.to_excel(output_path + "/uniform.xlsx", index=False)
    print(uniform_df)
    
    return uniform_df
