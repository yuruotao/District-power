# Coding: utf-8
# Handling missing data and imputation
import pandas as pd
import os
import missingno as msno
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates

def load_missing_value_visualization(input_df, save_path):
    """visualize the missing data in both bar and matrix format

    Args:
        input_df (dataframe): the dataframe containing raw data
        save_path (string): folder to save the plot

    Returns:
        None
    """
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    sns.set_theme(style="white")
    mpl.rcParams['font.family'] = 'Times New Roman'
    
    input_df = input_df.set_index('DATETIME')
    print(input_df)
    column_list = input_df.columns.values.tolist()

    missing_dir_matrix = save_path + "/matrix/"
    missing_dir_bar = save_path + "/bar/"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if not os.path.exists(missing_dir_matrix):
        os.makedirs(missing_dir_matrix)
        
    if not os.path.exists(missing_dir_bar):
        os.makedirs(missing_dir_bar)
    
    grouped_strings = {}

    # Iterate over the list of strings
    for string in column_list:
        # Extract 'a' and 'b' components
        a, b, _ = map(int, string.split('-'))
        # Get or create a list for the current 'a' and 'b' components
        group_key = (a, b)
        if group_key not in grouped_strings:
            grouped_strings[group_key] = []
        # Append the string to the list
        grouped_strings[group_key].append(string)

    # Sort the strings within each group
    for group_key, group_list in grouped_strings.items():
        grouped_strings[group_key] = sorted(group_list)

    # Create a list containing sorted lists of strings with the same 'a' and 'b' components
    result_list = grouped_strings.values()

    for list in result_list:
        sorted_list = sorted(list, key=lambda x: int(x.split('-')[2]))

        temp_df = input_df[sorted_list]
        index = list[0].split("-")
        temp_df.columns = temp_df.columns.str.split('-').str[-1]
        
        # Matrix plot
        ax = msno.matrix(temp_df, fontsize=10.5, figsize=(8, 6), label_rotation=0, freq="M")
        ax.tick_params(labelsize=10.5)
        plt.xlabel("Transformers", fontsize=10.5)
        plt.savefig(missing_dir_matrix + "City-" + index[0] + "-" + "District-" + index[1] +'-matrix.png', dpi=600)
        plt.close()
        
        # Bar plot
        ax = msno.bar(temp_df, fontsize=10.5, figsize=(8, 6), label_rotation=0)
        ax.tick_params(labelsize=10.5)
        plt.xlabel("Transformers", fontsize=10.5)
        plt.savefig(missing_dir_bar + "City-" + index[0] + "-" + "District-" + index[1] +'-bar.png', dpi=600)
        plt.close()
    
    return None

def transformer_missing_filter(meta_df, merged_df, threshold):
    """Delete the stations whose missing data percentage reach the threshold

    Args:
        meta_df (dataframe): dataframe containing the NCDC station meta data
        merged_df (merged_df): raw data merged_df
        threshold (float): threshold for deletion

    Returns:
        None
    """

    # Calculate percentage of missing values in each column
    missing_percentages = merged_df.isna().mean() * 100
    
    # Drop columns where the percentage of missing values exceeds the threshold
    columns_to_drop = missing_percentages[missing_percentages > threshold].index
    processed_df = merged_df.drop(columns=columns_to_drop)
    stations_higher_than_threshold = processed_df.columns.to_list()
    stations_higher_than_threshold.remove("DATETIME")

    filtered_meta_df = meta_df[meta_df['TRANSFORMER_ID'].isin(stations_higher_than_threshold)].reset_index(drop=True)

    return filtered_meta_df

def transformer_data_imputation(filtered_meta_df, merged_df):
    """impute the missing data

    Args:
        filtered_meta_df (dataframe): meta data of flow stations
        merged_df (dataframe): flow data
        engine (sqlalchemy_engine): engine to save the imputed flow data
        commit_flag (bool): whether to commit data to the database

    Returns:
        imputed dataframe
    """
    time_index = pd.date_range(start="2022-01-01 00:00:00", end="2023-11-11 23:00:00", freq="H")
    datetime_df = pd.DataFrame()
    datetime_df["DATETIME"] = time_index
    
    transformer_id_list = filtered_meta_df["TRANSFORMER_ID"].to_list()
    imputed_df = pd.DataFrame()
    for transformer_id in transformer_id_list:
        print(transformer_id)
        temp_df = merged_df[merged_df["TRANSFORMER_ID"] == transformer_id]
        temp_df = pd.merge(datetime_df, temp_df, on="DATETIME", how="left")
        
        datetime_column = temp_df["DATETIME"]
        temp_df = temp_df.drop(columns=["DATETIME", "TRANSFORMER_ID"])
        forward_df = temp_df.shift(-24)
        backward_df = temp_df.shift(24)
        average_values = (forward_df + backward_df) / 2
        
        temp_df = temp_df.copy()
        temp_df[temp_df.isna() & forward_df.notna() & backward_df.notna()] = average_values[temp_df.isna() & forward_df.notna() & backward_df.notna()]
        temp_df[temp_df.isna() & forward_df.notna() & backward_df.isna()] = forward_df[temp_df.isna() & forward_df.notna() & backward_df.isna()]
        temp_df[temp_df.isna() & backward_df.notna() & forward_df.isna()] = backward_df[temp_df.isna() & backward_df.notna() & forward_df.isna()]

        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        temp_df.set_index('DATETIME', inplace=True)
        # Set Datetime column as index
        for column in temp_df.columns:
            mean_value = temp_df[column].mean()
            # Fill NaN values with the mean
            temp_df[column].fillna(mean_value, inplace=True)

        temp_df = temp_df.reset_index()
        temp_df["TRANSFORMER_ID"] = transformer_id
        
        imputed_df = pd.concat([imputed_df, temp_df], axis=0)

    return imputed_df

def imputation(input_df, imputation_method, save_path, save_flag):
    """carry out the imputation for raw data with missing values

    Args:
        input_df (dataframe): the dataframe containing raw data
        imputation_method (string): specify the method of imputation
        save_path (string): specify the folder to save the imputed data
        save_flag (Bool): whether to save the imputed result

    Returns:
        dataframe: dataframe containing the imputed data
    """
    
    print("Imputation begin")
    datetime_column = input_df["DATETIME"]
    input_df = input_df.drop(columns=["DATETIME"])
    
    imputation_dir = save_path + "/"
    
    if not os.path.exists(imputation_dir):
        os.makedirs(imputation_dir)
        
    if imputation_method == "Linear":
        imputed_df = input_df.interpolate(method='linear')

    elif imputation_method == "Forward-Backward":
        forward_df = input_df.shift(-24)
        backward_df = input_df.shift(24)
        
        average_values = (forward_df + backward_df) / 2
        temp_df = input_df.copy()
        temp_df[temp_df.isna() & forward_df.notna() & backward_df.notna()] = average_values[temp_df.isna() & forward_df.notna() & backward_df.notna()]
        temp_df[temp_df.isna() & forward_df.notna() & backward_df.isna()] = forward_df[temp_df.isna() & forward_df.notna() & backward_df.isna()]
        temp_df[temp_df.isna() & backward_df.notna() & forward_df.isna()] = backward_df[temp_df.isna() & backward_df.notna() & forward_df.isna()]

        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        temp_df.set_index('DATETIME', inplace=True)
        # Set Datetime column as index

        for column in temp_df.columns:
            print(column)
            # Create a new DataFrame with day of the week and time of day as columns
            df_grouped = temp_df[column].reset_index()
            df_grouped['dayofweek'] = df_grouped['DATETIME'].dt.dayofweek
            df_grouped['time'] = df_grouped['DATETIME'].dt.time
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

        imputed_df = temp_df.reset_index(drop=True)

    elif imputation_method == "Forward":
        imputed_df = input_df.fillna(input_df.shift(-24))
        temp_df = imputed_df
        
        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        temp_df.set_index('DATETIME', inplace=True)
        # Set Datetime column as index

        for column in temp_df.columns:
            print(column)
            # Create a new DataFrame with day of the week and time of day as columns
            df_grouped = temp_df[column].reset_index()
            df_grouped['dayofweek'] = df_grouped['DATETIME'].dt.dayofweek
            df_grouped['time'] = df_grouped['DATETIME'].dt.time
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

        imputed_df = temp_df.reset_index(drop=True)
        
    elif imputation_method == "Backward":
        imputed_df = input_df.fillna(input_df.shift(24))
        temp_df = imputed_df
        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        temp_df.set_index('DATETIME', inplace=True)
        # Set Datetime column as index

        for column in temp_df.columns:
            print(column)
            # Create a new DataFrame with day of the week and time of day as columns
            df_grouped = temp_df[column].reset_index()
            df_grouped['dayofweek'] = df_grouped['DATETIME'].dt.dayofweek
            df_grouped['time'] = df_grouped['DATETIME'].dt.time
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

        imputed_df = temp_df.reset_index(drop=True)
    

    imputed_df = pd.concat([datetime_column, imputed_df], axis=1)
    if save_flag:
        imputed_df.to_excel(imputation_dir + "/" + "imputed_data_" + imputation_method + ".xlsx", index=False)
    else:
        pass
    
    return imputed_df

def lineplot_breaknans(data, break_at_nan=True, break_at_inf=True, **kwargs):
    """Make lineplot break at nans

    Args:
        data (dataframe): data to be plotted
        break_at_nan (bool, optional): whether to break at nan. Defaults to True.
        break_at_inf (bool, optional): whether to break at inf. Defaults to True.

    Raises:
        ValueError: dataframe do not contain any column

    Returns:
        ax
    """
    
    # Automatically detect the y column and use index as x
    if 'y' not in kwargs:
        columns = data.columns
        if len(columns) >= 1:
            kwargs['y'] = columns[0]
        else:
            raise ValueError("DataFrame must contain at least one column for y detection.")
    
    # Reset index to have a column for the x-axis
    data_reset = data.reset_index()
    kwargs['x'] = data_reset.columns[0]

    # Create a cumulative sum of NaNs and infs to use as units
    cum_num_nans_infs = np.zeros(len(data_reset))
    if break_at_nan: cum_num_nans_infs += np.cumsum(np.isnan(data_reset[kwargs['y']]))
    if break_at_inf: cum_num_nans_infs += np.cumsum(np.isinf(data_reset[kwargs['y']]))

    # Plot using seaborn's lineplot
    ax = sns.lineplot(data=data_reset, **kwargs, units=cum_num_nans_infs, estimator=None)  # estimator must be None when specifying units
    return ax

def imputation_visualization(raw_data_df, start_time, end_time, method_list, column, output_path):
    """Visualize the imputation methods

    Args:
        raw_data_df (dataframe): dataframe containing raw flow data
        start_time (datetime): start time for plot
        end_time (datetime): end time for plot
        method_list (list): methods to be plotted
        column (string): station for visualization
        output_path (string): path to save the output figure

    Returns:
        None
    """
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    sns.set_theme(style="white")
    mpl.rcParams['font.family'] = 'Times New Roman'
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    raw_data_df = raw_data_df[["DATETIME", column]]
    raw_data_df = raw_data_df.loc[(raw_data_df['DATETIME'] >= start_time) & (raw_data_df['DATETIME'] <= end_time)]
    raw_data_df = raw_data_df.rename(columns={column:"Raw"})
    raw_data_df = raw_data_df.fillna(value=np.nan)
    raw_data_df.replace(to_replace=[None], value=np.nan, inplace=True)
    
    time_index = pd.date_range(start=start_time, end=end_time, freq="h")
    # Create a DataFrame with the time series column
    time_series_df = pd.DataFrame({'DATETIME': time_index})
    for method in method_list:
        temp_df = pd.read_excel("./result/load_imputation/imputed_data_" + method +".xlsx")
        temp_df = temp_df[["DATETIME", column]]
        temp_df = temp_df.loc[(temp_df['DATETIME'] >= start_time) & (temp_df['DATETIME'] <= end_time)]
        temp_df = temp_df.rename(columns={column:method})
        time_series_df = pd.merge(time_series_df, temp_df, on='DATETIME', how="left")
        
    time_series_df = pd.merge(time_series_df, raw_data_df, on='DATETIME', how="left")
    time_series_df = time_series_df.set_index("DATETIME")
    
    plt.figure(figsize=(8, 5))
    ax = lineplot_breaknans(data=time_series_df, y="Raw", markers=True, linewidth=1.5, break_at_nan=True)
    
    columns_to_plot = [col for col in time_series_df.columns if col != "Raw" and col != "Forward-Backward"]
    temp_time_series_df = time_series_df[columns_to_plot]
    sns.lineplot(data=temp_time_series_df, ax=ax, markers=True, linewidth=1.5)
    
    missing_mask = time_series_df['Raw'].isna().values.astype(int)
    ax.set_xlim(time_series_df.index[0], time_series_df.index[-1])
    ax.pcolorfast(ax.get_xlim(), ax.get_ylim(),
                  missing_mask[np.newaxis], cmap='Blues', alpha=0.2)
    
    if "Forward-Backward" in time_series_df.columns:
        sns.lineplot(data=time_series_df["Forward-Backward"], ax=ax, color='#000000', linewidth=1.5, label="Forward-Backward")
    
    plt.rc('legend', fontsize=10.5)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    
    # Set the date format on the x-axis to show minutes
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

    # Put a legend below current axis
    ax.legend(loc='lower left', mode="expand", bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=4, frameon=False)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.tick_params(labelsize=10.5)
    plt.xlabel("Time (hour)", fontsize=10.5)
    plt.ylabel("Power (kW)", fontsize=10.5)
        
    #plt.tight_layout()
    plt.savefig(output_path + "imputation_methods.png", dpi=600)
    plt.close()
    
    return None

