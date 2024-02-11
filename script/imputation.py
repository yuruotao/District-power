import pandas as pd
import os
from datetime import timedelta, datetime
from fancyimpute import BiScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.arima import ARIMA
import numpy as np


def imputation(input_df, imputation_method, save_path):
    datetime_column = input_df["Datetime"]
    input_df = input_df.drop(columns=["Datetime"])
    
    imputation_dir = save_path + "/"
    
    if not os.path.exists(imputation_dir):
        os.makedirs(imputation_dir)

    if imputation_method == "MICE":
        # Initialize IterativeImputer with RandomForestRegressor as the estimator
        imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=2), 
                                   max_iter=10, tol=0.001)
        # Impute missing values
        imputed_data = imputer.fit_transform(input_df)
        # Convert imputed data back to DataFrame
        imputed_df = pd.DataFrame(imputed_data, index=input_df.index, columns=input_df.columns)
        
    elif imputation_method == "BiScaler":
        # Perform BiScaler imputation followed by IterativeImputer
        imputer = BiScaler()  # BiScaler transformation
        df_imputed = pd.DataFrame(imputer.fit_transform(input_df.values), columns=input_df.columns, index=input_df.index)  # Transform DataFrame

        imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=2), 
                                   max_iter=10, tol=0.001)  # Imputation method after BiScaler
        imputed_data = imputer.fit_transform(df_imputed)  # Impute missing values

        # Convert the result back to a DataFrame
        imputed_df = pd.DataFrame(imputed_data, columns=input_df.columns, index=input_df.index)
        
    elif imputation_method == "FEDOT":
        from fedot.core.repository.tasks import Task, TaskTypesEnum
        from fedot.core.data.data import InputData
        from fedot.core.data.data_split import train_test_data_setup
        from fedot.core.data.multi_modal import MultiModalData
        from fedot.core.repository.dataset_types import DataTypesEnum
        from fedot.core.repository.quality_metrics_repository import MetricsRepository
        from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
        from fedot.core.pipelines.pipeline import Pipeline
        
        input_data = InputData.from_dataframe(input_df, task=Task(TaskTypesEnum.imputation))

        # Define a Fedot pipeline for imputation
        node_imputation = PrimaryNode('imputation', nodes_from=[PrimaryNode('ridge')])  # Example imputation method
        node_main = SecondaryNode('ridge', nodes_from=[node_imputation])
        pipeline = Pipeline(node_main)

        # Fit the pipeline
        pipeline.fit(input_data)

        # Transform the data
        imputed_data = pipeline.predict(input_data)
        
        # Convert imputed data back to DataFrame
        imputed_df = imputed_data.features
          
    elif imputation_method == "Forward-Backward":
        forward_df = input_df.shift(-7*24)
        backward_df = input_df.shift(-7*24)
        for column in input_df.columns:
            # Create a mask to identify positions where either df2 or df3 has a non-NaN value
            mask = ~(forward_df[column].isna() & backward_df[column].isna())
            # Calculate the sum of corresponding values in df2 and df3, ignoring NaN values
            sum_values = forward_df[column].add(backward_df[column], fill_value=0)
            # Count the number of non-missing values in df2 and df3
            count_values = forward_df[column].notna().astype(int) + backward_df[column].notna().astype(int)
            # Calculate the average, using available values if one of the values is missing
            average_values = sum_values / count_values.where(count_values > 0, 1)
            # Fill missing values in df1 with the calculated average only where the mask is True
            input_df[column] = input_df[column].where(mask, average_values)

        temp_df = input_df
        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        temp_df.set_index('Datetime', inplace=True)
        # Set Datetime column as index

        for column in temp_df.columns:
            print(column)
            # Create a new DataFrame with day of the week and time of day as columns
            df_grouped = temp_df[column].reset_index()
            df_grouped['dayofweek'] = df_grouped['Datetime'].dt.dayofweek
            df_grouped['time'] = df_grouped['Datetime'].dt.time
            # Group by day of the week and time of day, then calculate the mean
            mean_values = df_grouped.groupby(['dayofweek', 'time'])[column].mean().reset_index()
            # Map the mean values to the corresponding missing values
            temp_df[column] = temp_df.index.map(lambda x: mean_values.loc[(mean_values['dayofweek'] == x.dayofweek) & (mean_values['time'] == x.time()), column].values[0])
        temp_df = temp_df.reset_index()
        imputed_df = temp_df.drop(columns=["Datetime"])

    elif imputation_method == "Forward":
        imputed_df = input_df.fillna(input_df.shift(-7*24))
        temp_df = imputed_df
        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        temp_df.set_index('Datetime', inplace=True)
        # Set Datetime column as index

        for column in temp_df.columns:
            print(column)
            # Create a new DataFrame with day of the week and time of day as columns
            df_grouped = temp_df[column].reset_index()
            df_grouped['dayofweek'] = df_grouped['Datetime'].dt.dayofweek
            df_grouped['time'] = df_grouped['Datetime'].dt.time
            # Group by day of the week and time of day, then calculate the mean
            mean_values = df_grouped.groupby(['dayofweek', 'time'])[column].mean().reset_index()
            # Map the mean values to the corresponding missing values
            temp_df[column] = temp_df.index.map(lambda x: mean_values.loc[(mean_values['dayofweek'] == x.dayofweek) & (mean_values['time'] == x.time()), column].values[0])
        temp_df = temp_df.reset_index()
        imputed_df = temp_df.drop(columns=["Datetime", "index"])
        
    elif imputation_method == "Backward":
        imputed_df = input_df.fillna(input_df.shift(7*24))
        temp_df = input_df
        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        temp_df.set_index('Datetime', inplace=True)
        # Set Datetime column as index

        for column in temp_df.columns:
            print(column)
            # Create a new DataFrame with day of the week and time of day as columns
            df_grouped = temp_df[column].reset_index()
            df_grouped['dayofweek'] = df_grouped['Datetime'].dt.dayofweek
            df_grouped['time'] = df_grouped['Datetime'].dt.time
            # Group by day of the week and time of day, then calculate the mean
            mean_values = df_grouped.groupby(['dayofweek', 'time'])[column].mean().reset_index()
            # Map the mean values to the corresponding missing values
            temp_df[column] = temp_df.index.map(lambda x: mean_values.loc[(mean_values['dayofweek'] == x.dayofweek) & (mean_values['time'] == x.time()), column].values[0])
        temp_df = temp_df.reset_index()
        imputed_df = temp_df.drop(columns=["Datetime"])
    
    elif imputation_method == "Average":
        temp_df = input_df
        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        temp_df.set_index('Datetime', inplace=True)
        # Set Datetime column as index

        for column in temp_df.columns:
            print(column)
            # Create a new DataFrame with day of the week and time of day as columns
            df_grouped = temp_df[column].reset_index()
            df_grouped['dayofweek'] = df_grouped['Datetime'].dt.dayofweek
            df_grouped['time'] = df_grouped['Datetime'].dt.time
            # Group by day of the week and time of day, then calculate the mean
            mean_values = df_grouped.groupby(['dayofweek', 'time'])[column].mean().reset_index()
            # Map the mean values to the corresponding missing values
            temp_df[column] = temp_df.index.map(lambda x: mean_values.loc[(mean_values['dayofweek'] == x.dayofweek) & (mean_values['time'] == x.time()), column].values[0])
        temp_df = temp_df.reset_index()
        imputed_df = temp_df.drop(columns=["Datetime"])
        

    print(imputed_df)
    imputed_df = pd.concat([datetime_column, imputed_df], axis=1)
    imputed_df.to_excel(imputation_dir + "/" + "imputed_data_" + imputation_method + ".xlsx", index=False)
    
    return imputed_df

def imputation_visualization(start_time, end_time, method_list, column, save_path):
    
    time_index = pd.date_range(start=start_time, end=end_time, freq="h")
    # Create a DataFrame with the time series column
    time_series_df = pd.DataFrame({'Datetime': time_index})
    for method in method_list:
        temp_df = pd.read_excel("./result/imputation/imputed_data_" + method +".xlsx")
        temp_df = temp_df[["Datetime", column]]
        temp_df = temp_df.loc[(temp_df['Datetime'] >= start_time) & (temp_df['Datetime'] <= end_time)]
        temp_df = temp_df.rename({column:method})
        time_series_df = pd.merge(time_series_df, temp_df, on='Datetime', how="left")
    
    
        
    
    return None