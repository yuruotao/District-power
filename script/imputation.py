import pandas as pd
import os
from datetime import timedelta, datetime
from fancyimpute import BiScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sktime.forecasting.arima import ARIMA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from autoimpute.imputations import SingleImputer, MultipleImputer
import numpy as np
import seaborn as sns


def imputation(input_df, imputation_method, save_path):
    """carry out the imputation for raw data with missing values

    Args:
        input_df (dataframe): the dataframe containing raw data
        imputation_method (string): specify the method of imputation
        save_path (string): specify the folder to save the imputed data

    Returns:
        dataframe: dataframe containing the imputed data
    """
    
    print("Imputation begin")
    datetime_column = input_df["Datetime"]
    input_df = input_df.drop(columns=["Datetime"])
    
    imputation_dir = save_path + "/"
    
    if not os.path.exists(imputation_dir):
        os.makedirs(imputation_dir)
        
    if imputation_method == "Linear":
        imputed_df = input_df.interpolate(method='linear')
        imputed_df = imputed_df.drop(columns=["index"])

    if imputation_method == "MICE":
        # Initialize IterativeImputer with RandomForestRegressor as the estimator
        imputer = IterativeImputer(estimator=RandomForestRegressor())
        # Impute missing values
        imputed_data = imputer.fit_transform(input_df)
        # Convert imputed data back to DataFrame
        imputed_df = pd.DataFrame(imputed_data, index=input_df.index, columns=input_df.columns)
        
    elif imputation_method == "BiScaler":
        # Perform BiScaler imputation followed by IterativeImputer
        imputer = BiScaler()  # BiScaler transformation
        df_imputed = pd.DataFrame(imputer.fit_transform(input_df.values), columns=input_df.columns, index=input_df.index)  # Transform DataFrame

        imputer = IterativeImputer(estimator=RandomForestRegressor())  # Imputation method after BiScaler
        imputed_data = imputer.fit_transform(df_imputed)  # Impute missing values

        # Convert the result back to a DataFrame
        imputed_df = pd.DataFrame(imputed_data, columns=input_df.columns, index=input_df.index)
        
    elif imputation_method == "AutoML":
        # Create a SingleImputer object
        imputer = MultipleImputer()

        # Fit the imputer to your data and transform it
        imputed_data = imputer.fit_transform(input_df)
        imputed_data_list = list(imputed_data)

        # Access individual imputed DataFrames
        for idx, df_imputed in enumerate(imputed_data_list):
            print(f"Imputed DataFrame {idx+1}:")
            print(df_imputed)
            
          
    elif imputation_method == "Forward-Backward":
        forward_df = input_df.shift(-7*24)
        backward_df = input_df.shift(-7*24)
        
        # Assuming df1 and df2 are your DataFrames
        result_df = forward_df.copy()  # Initialize result_df with df2's values

        # Divide by 2 where both elements are not NaN
        result_df[(backward_df.notna()) & (forward_df.notna())] = (backward_df[(backward_df.notna()) & (forward_df.notna())] + forward_df[(backward_df.notna()) & (forward_df.notna())]) / 2

        # Fill NaN values conditionally based on the presence of NaNs in backward_df and forward_df
        result_df[backward_df.notna()] = backward_df[backward_df.notna()]  # Fill non-NaN values from backward_df to result_df
        temp_df = input_df.fillna(result_df)

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
            
            # Function to fill missing values in a column based on datetime index
            def fill_missing_values(index_value, column, mean_values):
                if pd.isnull(temp_df.loc[index_value, column]):
                    # Find the corresponding mean value based on datetime index
                    mean_value = mean_values.loc[(mean_values['dayofweek'] == index_value.dayofweek) & (mean_values['time'] == index_value.time()), column].values[0]
                    return mean_value
                else:
                    return temp_df.loc[index_value, column]
            temp_df[column] = temp_df.apply(lambda row: fill_missing_values(row.name, column, mean_values), axis=1)

        temp_df = temp_df.reset_index()
        imputed_df = temp_df.drop(columns=["Datetime", "index"])

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
            
            # Function to fill missing values in a column based on datetime index
            def fill_missing_values(index_value, column, mean_values):
                if pd.isnull(temp_df.loc[index_value, column]):
                    # Find the corresponding mean value based on datetime index
                    mean_value = mean_values.loc[(mean_values['dayofweek'] == index_value.dayofweek) & (mean_values['time'] == index_value.time()), column].values[0]
                    return mean_value
                else:
                    return temp_df.loc[index_value, column]
            temp_df[column] = temp_df.apply(lambda row: fill_missing_values(row.name, column, mean_values), axis=1)

        temp_df = temp_df.reset_index()
        imputed_df = temp_df.drop(columns=["Datetime", "index"])
        
    elif imputation_method == "Backward":
        imputed_df = input_df.fillna(input_df.shift(7*24))
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
            
            # Function to fill missing values in a column based on datetime index
            def fill_missing_values(index_value, column, mean_values):
                if pd.isnull(temp_df.loc[index_value, column]):
                    # Find the corresponding mean value based on datetime index
                    mean_value = mean_values.loc[(mean_values['dayofweek'] == index_value.dayofweek) & (mean_values['time'] == index_value.time()), column].values[0]
                    return mean_value
                else:
                    return temp_df.loc[index_value, column]
            temp_df[column] = temp_df.apply(lambda row: fill_missing_values(row.name, column, mean_values), axis=1)

        temp_df = temp_df.reset_index()
        imputed_df = temp_df.drop(columns=["Datetime", "index"])
        
    print(imputed_df)
    imputed_df = pd.concat([datetime_column, imputed_df], axis=1)
    imputed_df.to_excel(imputation_dir + "/" + "imputed_data_" + imputation_method + ".xlsx", index=False)
    
    return imputed_df

def imputation_visualization(start_time, end_time, method_list, column, save_path):
    
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
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