# Coding: utf-8
# Calculate the correlation between weather and power, as well as the extreme weather
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
from scipy.stats import pearsonr

def weather_missing_filter(meta_df, merged_df, threshold):
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
    stations_higher_than_threshold.remove('DATETIME')
    
    filtered_meta_df = meta_df[meta_df['STATION_ID'].isin(stations_higher_than_threshold)].reset_index(drop=True)

    return filtered_meta_df

def NCDC_weather_data_imputation(filtered_meta_df, merged_df):
    """Reformat and impute the missing data of weather data
    Add relative humidity "RH" to the dataframe

    Args:
        filtered_meta_df (dataframe): meta dataframe
        merged_df (dataframe): dataframe containing merged weather data

    Returns:
        None
    """

    station_id_list = filtered_meta_df["STATION_ID"].to_list()
    imputed_df = pd.DataFrame()
    for station_id in station_id_list:
        print(station_id)
        temp_df = merged_df[merged_df["STATION_ID"] == station_id]
        datetime_column = temp_df['DATETIME']
        temp_df = temp_df.drop(columns=['DATETIME', "STATION_ID"])

        forward_df = temp_df.shift(-1)
        backward_df = temp_df.shift(1)
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
        temp_df["STATION_ID"] = station_id
        print(temp_df)
        imputed_df = pd.concat([imputed_df, temp_df], axis=0)

    return imputed_df

def weather_correlation(input_df, output_path, city_num):
    """Calculate the correlation among weather factors and plot the result

    Args:
        input_df (dataframe): contain the weather data
        output_path (string): path to save the plot
        city_num (string): corresponding city number

    Returns:
        None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    sns.set_theme(style="white")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    input_df =input_df.rename(columns = {"TEMP":"Temperature(C)",
                                                          "DEWP":"Dew Point(C)", 
                                                          #"RH":"Relative Humidity(%)",
                                                          #"PRCP":"Precipitation(m)",
                                                          "LOAD":"Power(kW)"
                                                          })
    
    # Plot the correlation
    mpl.rc('xtick', labelsize=10.5)
    mpl.rc('ytick', labelsize=10.5)
    mpl.rcParams["axes.labelsize"] = 10.5
    plt.rc('legend', fontsize=10.5)
    
    def corrfunc(x, y, **kwds):
        cmap = kwds['cmap']
        norm = kwds['norm']
        ax = plt.gca()
        ax.tick_params(bottom=False, top=False, left=False, right=False, axis="both", which="major", labelsize=10.5)
        sns.despine(ax=ax, bottom=True, top=True, left=True, right=True)
        r, p = pearsonr(x, y)
        facecolor = cmap(norm(r))
        ax.set_facecolor(facecolor)
        lightness = (max(facecolor[:3]) + min(facecolor[:3]) ) / 2
        # Correlation number on the plot
        ax.annotate(f"{r:.2f}\n({p:.2g})", xy=(.5, .5), xycoords=ax.transAxes,
                color='white' if lightness < 0.7 else 'black', size=26, ha='center', va='center')

    plt.figure(figsize=(8, 8))
    g = sns.PairGrid(input_df)
    g.map_lower(plt.scatter, s=22)
    g.map_diag(sns.histplot, kde=False)
    g.map_upper(corrfunc, cmap=plt.get_cmap('crest'), norm=plt.Normalize(vmin=0, vmax=1))
    
    # Adjust label size for all axes
    for ax in g.axes.flatten():
        ax.tick_params(axis='both', which='major', labelsize=10.5)
        ax.get_yaxis().set_label_coords(-0.25, 0.5)
    
    plt.tight_layout(rect=[0.02, 0, 1, 1])
    plt.savefig(output_path + "/correlation_" + city_num + ".png", dpi=600)
    plt.close()

    return None

def holiday_plot(input_df, city, weather_df, start_time, end_time, output_path):
    """Plot the extreme weather

    Args:
        input_df (dataframe): contain the power data
        city (string): city to plot (all)
        weather_df (dataframe): extreme weather data
        start_time (string): the start date of extreme weather
        end_date (string): the end date of extreme weather
        output_path (string): path to save the plot

    Returns:
        None
    """
    sns.set_theme(style="white")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    hourly_datetime = pd.date_range(start=weather_df['DATETIME'].min(), end=weather_df['DATETIME'].max() + pd.Timedelta(days=1) - pd.Timedelta(hours=1), freq='H')
    repeated_values = {col: weather_df[col].repeat(24).reset_index(drop=True) for col in weather_df.columns if col != 'DATETIME'}
    weather_df = pd.DataFrame({
        'DATETIME': hourly_datetime,
        **repeated_values})
    
    time_index = pd.date_range(start=start_time, end=end_time, freq="h")
    # Create a DataFrame with the time series column
    time_series_df = pd.DataFrame({'DATETIME': time_index})
    weather_df = pd.merge(time_series_df, weather_df, on='DATETIME', how="left")
    # Filter out missing values from weather_df
    weather_df.replace(to_replace=[""], value=np.nan, inplace=True)
    weather_df.replace(to_replace=[None], value=np.nan, inplace=True)
    weather_df_filtered = weather_df.fillna("None")
    
    time_series_df = pd.merge(time_series_df, input_df, on='DATETIME', how="left")
    time_series_df = pd.merge(time_series_df, weather_df_filtered, on='DATETIME', how="left")
    time_series_df = time_series_df.set_index('DATETIME')
    
    event_colors = {'New year':                     '#641220', 
                    'Dragon Boat Festival':         '#3fa34d', 
                    'International Labor Day':      '#22333b',
                    'Mid-autumn festival':          '#fdc500',
                    'National Day':                 '#e09f3e',
                    'Qingming':                     '#8b949a',
                    'Spring Festival':              '#e01e37',
                    'None':                         "#FFFFFF"
                    }
    

    fig, ax = plt.subplots(figsize=(14, 6), layout='constrained')
    ax.tick_params(axis='both', which='major', labelsize=10.5)
    ax.plot(time_series_df.index, time_series_df['LOAD'], color='#274c77')
    
    mpl.rc('xtick', labelsize=10.5)
    mpl.rc('ytick', labelsize=10.5)
    plt.rc('legend', fontsize=10.5)
    # Set background color according to Event values
    for event, color in event_colors.items():
        subset = time_series_df[time_series_df['HOLIDAY'] == event]
        print(event)
        
        dfs = []
        start_idx = subset.index[0]
        end_idx = None
        
        for idx in subset.index[1:]:
            if (idx - start_idx).days >= 1:
                # End of current part found
                dfs.append(subset.loc[start_idx:end_idx])
                start_idx = idx
                end_idx = None
            else:
                # Continuation of current part
                end_idx = idx

        # Add the last part of the DataFrame
        if end_idx is not None:
            dfs.append(subset.loc[start_idx:end_idx])
        
        df_num = 0
        for group_df in dfs:
            if event == "None":
                ax.axvspan(group_df.index[0], group_df.index[-1], alpha=0, edgecolor='none')
            else:
                if df_num == 0:
                    ax.axvspan(group_df.index[0], group_df.index[-1], facecolor=color, alpha=0.5, edgecolor='none', label=str(event))
                    df_num = df_num + 1
                else:
                    ax.axvspan(group_df.index[0], group_df.index[-1], facecolor=color, alpha=0.5, edgecolor='none')
                    df_num = df_num + 1

    ax.set_xlim(time_series_df.index.min(), time_series_df.index.max())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set(xlabel="", ylabel="")
    
    plt.xlabel("Time", fontsize=10.5)
    plt.ylabel("Power (kW)", fontsize=10.5)
    
    fig.legend(loc='outside center right', frameon=False)
    plt.savefig(output_path + "holiday_" + city + ".png", dpi=600)
    plt.close()
    
    return None

def extreme_weather_plot(input_df, city, weather_df, start_time, end_time, output_path):
    """Plot the extreme weather

    Args:
        input_df (dataframe): contain the power data
        city (string): city to plot (all)
        weather_df (dataframe): extreme weather data
        start_time (string): the start date of extreme weather
        end_date (string): the end date of extreme weather
        output_path (string): path to save the plot

    Returns:
        None
    """
    sns.set_theme(style="white")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    hourly_datetime = pd.date_range(start=weather_df['DATETIME'].min(), end=weather_df['DATETIME'].max() + pd.Timedelta(days=1) - pd.Timedelta(hours=1), freq='H')
    repeated_values = {col: weather_df[col].repeat(24).reset_index(drop=True) for col in weather_df.columns if col != 'DATETIME'}
    weather_df = pd.DataFrame({
        'DATETIME': hourly_datetime,
        **repeated_values})
    
    time_index = pd.date_range(start=start_time, end=end_time, freq="h")
    # Create a DataFrame with the time series column
    time_series_df = pd.DataFrame({'DATETIME': time_index})
    weather_df = pd.merge(time_series_df, weather_df, on='DATETIME', how="left")
    # Filter out missing values from weather_df
    weather_df.replace(to_replace=[""], value=np.nan, inplace=True)
    weather_df.replace(to_replace=[None], value=np.nan, inplace=True)
    weather_df_filtered = weather_df.fillna("None")
    
    time_series_df = pd.merge(time_series_df, input_df, on='DATETIME', how="left")
    time_series_df = pd.merge(time_series_df, weather_df_filtered, on='DATETIME', how="left")
    time_series_df = time_series_df.set_index("DATETIME")
    
    event_colors = {'Hot weather':                  '#bc4749', 
                    'Severe convective weather':    '#4f000b', 
                    'Cold wave':                    '#5a189a',
                    'Dragon-boat rain':             '#00a6fb',
                    'Drought':                      '#e09f3e',
                    'Excessive flooding':           '#003554',
                    'Extreme rainfall':             '#0582ca',
                    'Rainstorm':                    '#006494',
                    'Tropical Storm Sanba':         '#ff6700',
                    'Typhoon Chaba':                '#4f772d',
                    'Typhoon Haikui':               '#4f772d',
                    'None':                         "#FFFFFF"}
    

    fig, ax = plt.subplots(figsize=(14, 6), layout='constrained')
    ax.tick_params(axis='both', which='major', labelsize=10.5)
    ax.plot(time_series_df.index, time_series_df['LOAD'], color='#274c77')

    mpl.rc('xtick', labelsize=10.5)
    mpl.rc('ytick', labelsize=10.5)
    plt.rc('legend', fontsize=10.5)
    
    for event, color in event_colors.items():
        subset = time_series_df[time_series_df["HAZARD"] == event]
        print(event)
        
        if not subset.empty:
            dfs = []
            start_idx = subset.index[0]
            end_idx = None
        
            for idx in subset.index[1:]:
                if (idx - start_idx).days > 1:
                    # End of current part found
                    dfs.append(subset.loc[start_idx:end_idx])
                    start_idx = idx
                    end_idx = None
                else:
                    # Continuation of current part
                    end_idx = idx

            df_num = 0
            for group_df in dfs:
                if event == "None":
                    ax.axvspan(group_df.index[0], group_df.index[-1], alpha=0, edgecolor='none')
                else:
                    if df_num == 0:
                        ax.axvspan(group_df.index[0], group_df.index[-1], facecolor=color, alpha=0.5, edgecolor='none', label=str(event))
                        df_num = df_num + 1
                    else:
                        ax.axvspan(group_df.index[0], group_df.index[-1], facecolor=color, alpha=0.5, edgecolor='none')
                        df_num = df_num + 1
    

    fig.legend(loc='outside center right', frameon=False)

    ax.set_xlim(time_series_df.index.min(), time_series_df.index.max())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)        
    ax.set(xlabel="", ylabel="")
    
    plt.xlabel("Time", fontsize=10.5)
    plt.ylabel("Power (kW)", fontsize=10.5)

    plt.savefig(output_path + "extreme_weather_" + city + ".png", dpi=600)
    plt.close()
    
    return None

def extreme_weather_city_plot(input_df, city, weather_df, start_time, end_time, output_path):
    """Plot extreme weather load profile for each city

    Args:
        input_df (dataframe): contain the power data
        city (string): city to plot
        weather_df (dataframe): extreme weather data
        start_time (string): the start date of extreme weather
        end_date (string): the end date of extreme weather
        output_path (string): path to save the plot

    Returns:
        None
    """
    sns.set_theme(style="white")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    time_index = pd.date_range(start=start_time, end=end_time, freq="h")
    # Create a DataFrame with the time series column
    time_series_df = pd.DataFrame({'DATETIME': time_index})
    weather_df = pd.merge(time_series_df, weather_df, on='DATETIME', how="left")
    # Filter out missing values from weather_df
    weather_df.replace(to_replace=[""], value=np.nan, inplace=True)
    weather_df.replace(to_replace=[None], value=np.nan, inplace=True)
    weather_df_filtered = weather_df.fillna("None")
    
    time_series_df = pd.merge(time_series_df, input_df, on='DATETIME', how="left")
    time_series_df = pd.merge(time_series_df, weather_df_filtered, on='DATETIME', how="left")
    time_series_df = time_series_df.set_index("DATETIME")
    
    event_colors = {#"HEAT_INDEX_Caution":                   "#ad2831",
                    "HEAT_INDEX_EXTREME_CAUTION":           "#800e13",
                    "HEAT_INDEX_DANGER":                    "#640d14",
                    "HEAT_INDEX_EXTREME_DANGER":            "#38040e",
                    #"WIND_CHILL_VERY_COLD":                 "#0096c7",
                    #"WIND_CHILL_FROSTBITE_DANGER":          "#023e8a",
                    #"WIND_CHILL_GREAT_FROSTBITE_DANGER":    "#03045e",
                    "HIGH_TEMPERATURE": "#e5383b",
                    "LOW_TEMPERATURE": "#192bc2",
                    "HIGH_HUMIDITY": "#007f5f",
                    #"WIND_LEVEL_0": "#e0aaff",
                    #"WIND_LEVEL_1": "#c77dff",
                    #"WIND_LEVEL_2": "#9d4edd",
                    #"WIND_LEVEL_3": "#7b2cbf",
                    #"WIND_LEVEL_4": "#5a189a",
                    "WIND_LEVEL_5": "#e0aaff",
                    "WIND_LEVEL_6": "#c77dff",
                    "WIND_LEVEL_7": "#9d4edd",
                    "WIND_LEVEL_8": "#7b2cbf",
                    "WIND_LEVEL_9": "#5a189a",
                    #"WIND_LEVEL_10": "#55a630",
                    #"WIND_LEVEL_11": "#2b9348",
                    #"WIND_LEVEL_12": "#007f5f",
                    "PRECIPITATION_50": "#d5c7bc",
                    "PRECIPITATION_100": "#785964",
                    }
    

    fig, ax = plt.subplots(figsize=(14, 6), layout='constrained')
    ax.tick_params(axis='both', which='major', labelsize=10.5)
    ax.plot(time_series_df.index, time_series_df['LOAD'], color='#274c77')

    mpl.rc('xtick', labelsize=10.5)
    mpl.rc('ytick', labelsize=10.5)
    plt.rc('legend', fontsize=10.5)
    
    for event, color in event_colors.items():
        subset = time_series_df[time_series_df[event] == 1]
        print(event)
        
        if not subset.empty:
            dfs = []
            start_idx = subset.index[0]
            end_idx = None
        
            for idx in subset.index[1:]:
                if (idx - start_idx).days > 1:
                    # End of current part found
                    dfs.append(subset.loc[start_idx:end_idx])
                    start_idx = idx
                    end_idx = None
                else:
                    # Continuation of current part
                    end_idx = idx

            df_num = 0
            for group_df in dfs:
                if event == "None":
                    ax.axvspan(group_df.index[0], group_df.index[-1], alpha=0, edgecolor='none')
                else:
                    if df_num == 0:
                        ax.axvspan(group_df.index[0], group_df.index[-1], facecolor=color, alpha=0.5, edgecolor='none', label=str(event))
                        df_num = df_num + 1
                    else:
                        ax.axvspan(group_df.index[0], group_df.index[-1], facecolor=color, alpha=0.5, edgecolor='none')
                        df_num = df_num + 1
    

    fig.legend(loc='outside center right', frameon=False)

    ax.set_xlim(time_series_df.index.min(), time_series_df.index.max())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)        
    ax.set(xlabel="", ylabel="")
    
    plt.xlabel("Time", fontsize=10.5)
    plt.ylabel("Power (kW)", fontsize=10.5)

    plt.savefig(output_path + "extreme_weather_" + city + ".png", dpi=600)
    plt.close()
    
    return None

def extreme_normal_comparison_plot(input_df, weather_df, start_time, end_time, output_path):
    """Plot extreme weather load profile for each city

    Args:
        input_df (dataframe): contain the power data
        weather_df (dataframe): extreme weather data
        start_time (string): the start date of time range
        end_date (string): the end date of time range
        output_path (string): path to save the plot

    Returns:
        None
    """
    sns.set_theme(style="white")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    mpl.rcParams['font.family'] = 'Times New Roman'
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    time_index = pd.date_range(start=start_time, end=end_time, freq="h")
    # Create a DataFrame with the time series column
    time_series_df = pd.DataFrame({'DATETIME': time_index})
    weather_df = pd.merge(time_series_df, weather_df, on='DATETIME', how="left")
    # Filter out missing values from weather_df
    weather_df.replace(to_replace=[""], value=np.nan, inplace=True)
    weather_df.replace(to_replace=[None], value=np.nan, inplace=True)
    weather_df_filtered = weather_df.fillna("None")
    
    time_series_df = pd.merge(time_series_df, input_df, on='DATETIME', how="left")
    time_series_df = pd.merge(time_series_df, weather_df_filtered, on='DATETIME', how="left")
    time_series_df = time_series_df.set_index("DATETIME")
    
    event_colors = {#"HEAT_INDEX_Caution":                   "#ad2831",
                    "HEAT_INDEX_EXTREME_CAUTION":           "#800e13",
                    "HEAT_INDEX_DANGER":                    "#640d14",
                    "HEAT_INDEX_EXTREME_DANGER":            "#38040e",
                    #"WIND_CHILL_VERY_COLD":                 "#0096c7",
                    #"WIND_CHILL_FROSTBITE_DANGER":          "#023e8a",
                    #"WIND_CHILL_GREAT_FROSTBITE_DANGER":    "#03045e",
                    "HIGH_TEMPERATURE": "#e5383b",
                    "LOW_TEMPERATURE": "#192bc2",
                    "HIGH_HUMIDITY": "#007f5f",
                    #"WIND_LEVEL_0": "#e0aaff",
                    #"WIND_LEVEL_1": "#c77dff",
                    #"WIND_LEVEL_2": "#9d4edd",
                    #"WIND_LEVEL_3": "#7b2cbf",
                    #"WIND_LEVEL_4": "#5a189a",
                    "WIND_LEVEL_5": "#e0aaff",
                    "WIND_LEVEL_6": "#c77dff",
                    "WIND_LEVEL_7": "#9d4edd",
                    "WIND_LEVEL_8": "#7b2cbf",
                    "WIND_LEVEL_9": "#5a189a",
                    #"WIND_LEVEL_10": "#55a630",
                    #"WIND_LEVEL_11": "#2b9348",
                    #"WIND_LEVEL_12": "#007f5f",
                    #"PRECIPITATION_50": "#d5c7bc",
                    "PRECIPITATION_100": "#785964",
                    }
    
    alphabet_list = [chr(chNum) for chNum in list(range(ord('a'),ord('z')+1))]
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
    # Flatten the axes array for easy iteration
    axs = axs.flatten()
    
    event_num = 0
    for event, color in event_colors.items():
        subset = time_series_df[time_series_df[event] == 1]
        print(event)
        
        if not subset.empty:
            dfs = []
            start_idx = subset.index[0]
            end_idx = None
        
            found = 0
            for idx in subset.index[1:]:
                
                if found == 1:
                    #break
                    pass
                    
                if (idx - start_idx).days >= 1:
                    # End of current part found
                    print(start_idx)
                    if found == 0:
                        start_time = start_idx
                        end_time = start_idx + pd.Timedelta(days=1)
                        
                    dfs.append(subset.loc[start_idx:end_idx])
                    start_idx = idx
                    end_idx = None
                    found = 1
                else:
                    # Continuation of current part
                    end_idx = idx
            if dfs:
                ax = axs[event_num]
                
                if event == "HEAT_INDEX_EXTREME_CAUTION":
                    event = "Heat index extreme caution"
                    start_time = pd.Timestamp('2022-06-30 00:00:00')
                    end_time = start_time + pd.Timedelta(days=1)
                elif event == "HIGH_TEMPERATURE":
                    event = "High temperature"
                    start_time = pd.Timestamp('2022-10-04 00:00:00')
                    end_time = start_time + pd.Timedelta(days=1)
                elif event == "LOW_TEMPERATURE":
                    event = "Low temperature"
                    start_time = pd.Timestamp('2022-12-01 00:00:00')
                    end_time = start_time + pd.Timedelta(days=1)
                elif event == "HIGH_HUMIDITY":
                    event = "High humidity"
                    start_time = pd.Timestamp('2022-03-24 00:00:00')
                    end_time = start_time + pd.Timedelta(days=1)
                elif event == "WIND_LEVEL_5":
                    event = "Level 5 wind"
                    start_time = pd.Timestamp('2022-03-25 00:00:00')
                    end_time = start_time + pd.Timedelta(days=1)
                
                elif event == "PRECIPITATION_100":
                    event = "Precipitation 100mm"
                    start_time = pd.Timestamp('2022-06-04 00:00:00')
                    end_time = start_time + pd.Timedelta(days=1)
                
                extreme_df = time_series_df.loc[(time_series_df.index >= start_time) & (time_series_df.index < end_time)]
                
                extreme_df_before = time_series_df.loc[(time_series_df.index >= start_time - pd.Timedelta(days=1)) & 
                (time_series_df.index < end_time - pd.Timedelta(days=1))]
                extreme_df_before = extreme_df_before.shift(freq="24H")
                
                extreme_df_after = time_series_df.loc[(time_series_df.index >= start_time + pd.Timedelta(days=1)) & 
                (time_series_df.index < end_time + pd.Timedelta(days=1))]
                extreme_df_after = extreme_df_after.shift(freq="-24H")
                
                extreme_df_week_before = time_series_df.loc[(time_series_df.index >= start_time - pd.Timedelta(days=7)) & 
                (time_series_df.index <= end_time - pd.Timedelta(days=7))]
                extreme_df_week_before = extreme_df_week_before.shift(freq="168H")
                """
                extreme_df_week_average = time_series_df.loc[(time_series_df.index >= start_time - pd.Timedelta(days=3)) & 
                (time_series_df.index < end_time + pd.Timedelta(days=3))]
                extreme_df_week_average = extreme_df_week_average[["LOAD"]]

                # Group by time of day and calculate the mean for each time slot
                average_by_time_of_day = extreme_df_week_average.groupby(extreme_df_week_average.index.time).mean()
                extreme_df_week_average = pd.DataFrame({"DATETIME": extreme_df.index,
                                                        "LOAD": average_by_time_of_day["LOAD"]})
                extreme_df_week_average = extreme_df_week_average.set_index("DATETIME")
                print(extreme_df_week_average)
                """
    
                if event_num == 0:
                    ax.plot(extreme_df.index, extreme_df['LOAD'], color="#ba181b", label="Extreme", linewidth=1.5)
                    ax.plot(extreme_df_before.index, extreme_df_before['LOAD'], color='#274c77', label="Previous Day", linewidth=1.5)
                    #ax.plot(extreme_df_after.index, extreme_df_after['LOAD'], color='#fca311', label="Next Day", linewidth=1.5)
                    ax.plot(extreme_df_week_before.index, extreme_df_week_before['LOAD'], color='#0096c7', label="Same Day Last Week", linewidth=1.5)
                else:
                    ax.plot(extreme_df.index, extreme_df['LOAD'], color="#ba181b", linewidth=1.5)
                    ax.plot(extreme_df_before.index, extreme_df_before['LOAD'], color='#274c77', linewidth=1.5)
                    #ax.plot(extreme_df_after.index, extreme_df_after['LOAD'], color='#fca311', linewidth=1.5)
                    ax.plot(extreme_df_week_before.index, extreme_df_week_before['LOAD'], color='#0096c7', linewidth=1.5)
                
                ax.tick_params(top=False, bottom=True, left=False, right=False)
                ax.tick_params(which='both', direction='in', length=2)
                ax.set_title("(" + alphabet_list[event_num] + ") " + event, fontsize=10.5)
                ax.set_xlim(extreme_df.index.min(), extreme_df.index.max())
                ax.tick_params(axis='both', labelsize=10.5)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45) 
                ax.set(xlabel="", ylabel="")
                
                event_num = event_num + 1
    
    # Hide the empty subplots
    for ax in axs[6:]:
        ax.axis('off')

    # Adjust layout
    plt.tight_layout(rect=[0.02, 0.05, 1, 1])
    # Add ylabel to the entire figure
    fig.text(0.004, 0.5, 'Power (kW)', va='center', rotation='vertical', fontsize=10.5)
    fig.legend(loc='lower center', ncol=3, frameon=False, fontsize=10.5)
    # Show the plot
    plt.savefig(output_path + "extreme_weather_sub_all.png", dpi=600)
    plt.close()
    
    return None


if __name__ == "__main__":
    guilin_df = pd.read_excel("./guilin.xlsx")
    temp_extreme_weather_df = pd.read_excel("./temp_extreme.xlsx")
    extreme_normal_comparison_plot(guilin_df,
                                    temp_extreme_weather_df,
                                    "2022-01-01 00:00:00", '2023-10-31 23:00:00', 
                                    "./result/extreme_weather/")