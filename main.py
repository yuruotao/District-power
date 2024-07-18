# Coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import missingno as msno
import seaborn as sns
import matplotlib as mpl
import matplotlib.dates as mdates
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from utils.load_missing import *


if __name__ == "__main__":
    # 1. Initialization and import data from database
    time_index = pd.date_range(start="2022-01-01 00:00:00", end="2023-11-11 23:00:00", freq="H")
    datetime_df = pd.DataFrame()
    datetime_df["DATETIME"] = time_index
    
    db_address = 'sqlite:///./data/Transformer_DB/Transformer_DB.db'
    engine = create_engine(db_address)

    transformer_raw_query = 'SELECT * FROM transformer_raw'
    transformer_raw_df = pd.read_sql(transformer_raw_query, engine)
    transformer_raw_df = transformer_raw_df.astype({"DATETIME":"datetime64[ms]"})
    
    transformer_meta_query = 'SELECT * FROM transformer_meta'
    transformer_meta_df = pd.read_sql(transformer_meta_query, engine)
    ############################################################################################################
    # 2. Missing data, filter, imputation
    transformer_pivot_df = transformer_raw_df.pivot(index='DATETIME', columns='TRANSFORMER_ID', values='LOAD')
    transformer_pivot_df = datetime_df.merge(transformer_pivot_df, on='DATETIME', how='left')
    # Statistics for load data
    transformer_pivot_df.describe().to_excel("./result/load_summary.xlsx")
    
    # Load missing data
    missing_data_flag = False
    if missing_data_flag:
        load_missing_value_visualization(transformer_pivot_df, "./result/load_missing")

    # Filter by the percentage of missing data
    filtered_transformer_meta_df = transformer_missing_filter(transformer_meta_df, transformer_pivot_df, 30)
    
    # Imputation
    #imputed_transformer_df = transformer_data_imputation(filtered_transformer_meta_df, transformer_raw_df)

    # Imputation visualization
    imputation_visualization_flag = False
    if imputation_visualization_flag:
        single_transformer_df = transformer_pivot_df[["DATETIME", "0-0-0"]]
        imputation_methods = ["Linear", "Forward", "Backward", "Forward-Backward"]
        for method in imputation_methods:
            imputed_df = imputation(single_transformer_df, save_path="./result/load_imputation", imputation_method=method, save_flag=True)
            
        imputation_visualization(single_transformer_df, '2022-06-17 00:00:00', '2022-06-19 00:00:00', 
                                            ["Linear", "Forward", "Backward", "Forward-Backward"],
                                            "0-0-0",
                                            "./result/load_imputation/")
    ############################################################################################################
    # 3. Analysis
    # 3.1 Diversity factor plot
    