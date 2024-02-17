import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import calplot
import datetime
from scipy.stats import pearsonr

def average_load_profile(input_df, output_path):
    """Plot the load profile curve

    Args:
        input_df (dataframe): contain the data to be plotted
        output_path (string): path to save the plot

    Returns:
        None
    """
    
    sns.set_theme(style="whitegrid")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    start_time = '2022-01-01 00:00:00'
    end_time = '2023-11-10 08:00:00'
    temp_df = input_df.drop(["Datetime"], axis=1)
    
    input_df = input_df.loc[(input_df['Datetime'] >= start_time) & (input_df['Datetime'] <= end_time)]
    color_list = ["#669bbc", "#003049", "#780000", 
        "#0466c8", "#d90429",  "#0353a4", "#023e7d", "#002855", "#001845", "#001233", "#33415c", "#5c677d", "#7d8597", "#979dac", ]

    column_names = temp_df.columns
    subplot_num = 3
    fig, axes = plt.subplots(subplot_num, 1, figsize=(20, 7))
    for iter in range(subplot_num):

        sns.lineplot(data=input_df, x='Datetime', y=column_names[iter], ax=axes[iter], color=color_list[iter])
        axes[iter].set_title(column_names[iter])

        n = 1000  # Set the desired frequency of ticks
        ticks = input_df.iloc[::n, 0]  # Select every nth tick from the 'Date' column
        axes[iter].set_xticks(ticks)
        axes[iter].set(xlabel="", ylabel="")

            
    # Hide xticks for all subplots except the bottom one
    for ax in axes[:-1]:
        ax.xaxis.set_tick_params(which='both', bottom=False, top=False, labelbottom=False)

    # Display xticks only at the bottom subplot
    axes[-1].xaxis.set_ticks_position('both')  # Display ticks on both sides
    axes[-1].xaxis.set_tick_params(which='both', bottom=True, top=False)  # Only bottom ticks are visible
    
    
    # Rotate the x-tick labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout
    plt.tight_layout()
    # Show the plot
    plt.savefig(output_path + "city_load_profile.png", dpi=600)
    plt.close()
    
    return None

def diversity_factor_all(input_df, meta_df, output_path, type):
    """calculate the diversity factor for all transformers

    Args:
        input_df (dataframe): the dataframe containing all transformers' load profile
        meta_df (dataframe): the dataframe containing rated capacity for transformers
        output_path (string): path to store the output xlsx
        type (string): specify the definition of max_load. "Rated_capacity" for meta.xlsx reference

    Returns:
        list: containing the DF dataframe of input dataframe
        list: contain the name of dataframes
    """
    datetime_column = input_df["Datetime"]
    temp_df = input_df.drop(["Datetime"], axis=1)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Calculate total load for each hour by summing across all transformers
    total_load_per_hour = temp_df.sum(axis=1)
    print(total_load_per_hour)
    
    if type == "Rated_capacity":
        max_load = meta_df["YXRL"].sum()
        
    else:
        # Calculate maximum load across all hours
        max_load = total_load_per_hour.max()
    
    # Calculate diversity factor for each hour
    diversity_factor = total_load_per_hour / max_load
    diversity_factor_df = pd.DataFrame(diversity_factor, columns=['Diversity Factor'])
    diversity_factor_df = pd.concat([datetime_column, diversity_factor_df], axis=1)
    
    print(diversity_factor_df)
    diversity_factor_df.to_excel(output_path + "DF_all_transformers.xlsx", index=False)
    
    return [diversity_factor_df], ["all"]

def diversity_factor(input_df, meta_df, output_path, type):
    """calculate the diversity factor for all transformers in the same district

    Args:
        input_df (dataframe): the dataframe containing all transformers' load profile
        meta_df (dataframe): the dataframe containing rated capacity for transformers
        output_path (string): path to store the output xlsx
        type (string): specify the definition of max_load. "Rated_capacity" for meta.xlsx reference

    Returns:
        list: containing the DF dataframes for each district
        list: contain the name of dataframes
    """
    datetime_column = input_df["Datetime"]
    temp_df = input_df.drop(["Datetime"], axis=1)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Group columns by the first two parts of the column names
    grouped = temp_df.groupby(temp_df.columns.str.split('-', expand=True).map(lambda x: '-'.join(x[:2])), axis=1)
    # Extract sub-DataFrames with the same 'a' and 'b' values
    sub_dataframes = [sub_df for _, sub_df in grouped]

    return_list = []
    name_list = []
    # Print sub-DataFrames
    for i, sub_df in enumerate(sub_dataframes, 1):
        name = sub_df.columns[0].rsplit('-', 1)[0]
        name_list.append(name)
        print("Sub-DataFrame ", name)
        index = name.split("-")
        city_num = int(index[0])
        district_num = int(index[1])
        
        # Calculate total load for each hour by summing across all transformers
        total_load_per_hour = sub_df.sum(axis=1)
        
        # Calculate maximum load across all hours
        if type == "Rated_capacity":
            meta_df_temp = meta_df.loc[meta_df['City'] == city_num and meta_df['District'] == district_num]
            max_load = meta_df_temp["YXRL"].sum()
        else:
            # Calculate maximum load across all hours
            max_load = total_load_per_hour.max()
        
        # Calculate diversity factor for each hour
        diversity_factor = total_load_per_hour / max_load
        diversity_factor_df = pd.DataFrame(diversity_factor, columns=['Diversity Factor'])
        diversity_factor_df = pd.concat([datetime_column, diversity_factor_df], axis=1)
        return_list.append(diversity_factor_df)
        print(diversity_factor_df)
        diversity_factor_df.to_excel(output_path + "DF_" + name + ".xlsx", index=False)
    
    return return_list, name_list

def diversity_heatmap(input_df_list, name_list, output_path):
    """plot the heatmap of diversity factor

    Args:
        input_df_list (list): list containing dataframes of diversity factor
        name_list (list): contain the names of dataframes
        output_path (string): folder to store the heatmap plot

    Returns:
        None
    """
    sns.set_theme(style="whitegrid")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    start_time = '2022-01-01 00:00:00'
    end_time = '2023-11-10 08:00:00'
        
    for iter in range(len(input_df_list)):
        df = input_df_list[iter]
        name = name_list[iter]
        df = df.loc[(df['Datetime'] >= start_time) & (df['Datetime'] <= end_time)]
        
        df['hour'] = df['Datetime'].dt.hour
        df['date'] = df['Datetime'].dt.date
        df_pivot = df.pivot_table(index='date', columns='hour', values='Diversity Factor', aggfunc='mean')

        # Plot heatmap
        ax = sns.heatmap(df_pivot, cmap="crest", annot=False)
        # Get the current x-axis tick labels and positions
        xticklabels = ax.get_xticklabels()
        xtickpositions = ax.get_xticks()

        # Set the step size for displaying xticks (e.g., display every nth tick)
        step_size = 2

        # Filter the xtick labels and positions to show only every step_size-th tick
        filtered_xticklabels = [label.get_text() for i, label in enumerate(xticklabels) if i % step_size == 0]
        filtered_xtickpositions = [position for i, position in enumerate(xtickpositions) if i % step_size == 0]

        # Set the filtered xtick labels and positions
        ax.set_xticks(filtered_xtickpositions)
        ax.set_xticklabels(filtered_xticklabels, rotation=0)        
        ax.set(xlabel="", ylabel="")
        
        plt.tight_layout()
        plt.savefig(output_path + "DF_" + name + ".png", dpi=600)
        plt.close()
    
    return None

def year_DF_heatmap(input_df, meta_df, output_path, type):
    """Plot the Github style heatmap for diversity factor
    https://python.plainenglish.io/interactive-calendar-heatmaps-with-plotly-the-easieast-way-youll-find-5fc322125db7

    Args:
        input_df (dataframe): the dataframe containing all transformers' load profile
        meta_df (dataframe): the dataframe containing rated capacity for transformers
        output_path (string): path to store the output xlsx
        type (string): specify the definition of max_load. "Rated_capacity" for meta.xlsx reference

    Returns:
        None
    """
    datetime_column = input_df["Datetime"]
    temp_df = input_df.drop(["Datetime"], axis=1)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Calculate total load for each day by summing across all transformers
    total_load_per_day = temp_df.sum(axis=1)
    
    if type == "Rated_capacity":
        max_load = 24 * meta_df["YXRL"].sum()
    else:
        # Calculate maximum load across all hours
        max_load = total_load_per_day.max()

    # Calculate diversity factor for each day
    diversity_factor = total_load_per_day / max_load
    diversity_factor_df = pd.DataFrame(diversity_factor, columns=['Diversity Factor'])
    diversity_factor_df.reset_index()
    diversity_factor_df = pd.concat([datetime_column, diversity_factor_df], axis=1)

    diversity_factor_df.to_excel(output_path + "DF_daily_all_transformers.xlsx", index=False)
    diversity_factor_df = diversity_factor_df.set_index("Datetime")
    calplot.calplot(diversity_factor_df["Diversity Factor"], cmap="Blues")
    plt.savefig(output_path + "DF_daily_all.png", dpi=600)
    plt.close()
    
    return None 

def seasonality_decomposition(input_df, output_path, period_num, model):
    """decomposition the dataframe by seasonality

    Args:
        input_df (dataframe): the dataframe containing data
        output_path (string): path to save the figure
        period_num (int): the period length based on dataframe's resolution
        model (string): "additive" or multiplicative

    Returns:
        None
    """
    output_path = output_path + str(period_num) + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    start_time = '2022-01-01 00:00:00'
    end_time = '2022-12-31 23:00:00'
    input_df = input_df.loc[(input_df['Datetime'] >= start_time) & (input_df['Datetime'] <= end_time)]
    
    temp_df = input_df
    temp_df.set_index('Datetime', inplace=True)
    for column in temp_df.columns:
        print(column)
        plot_df = temp_df[[column]]
        check = (plot_df[column] == 0).all()
        if (not check):
            plot_df[column] = plot_df[column].apply(lambda x : x if x > 0 else 0)
            plot_df.replace(to_replace=0, value=1e-6, inplace=True)
            
            # Perform seasonal decomposition
            result = seasonal_decompose(plot_df, model=model, period=period_num)  # Adjust period as needed

            # Create a Matplotlib figure and axes
            fig, axes = plt.subplots(4, 1, figsize=(20, 8))

            # Plot the original time series
            axes[0].plot(plot_df, label='Original', color="#023e8a")
            axes[0].legend()

            # Plot the trend component
            axes[1].plot(result.trend, label='Trend', color="#0077b6")
            axes[1].legend()

            # Plot the seasonal component
            axes[2].plot(result.seasonal, label='Seasonal', color='#03045e')
            axes[2].legend()
        
            # Plot the residual component
            axes[3].plot(result.resid, label='Residual', color='#780000')
            axes[3].legend()
        
            # Adjust layout
            plt.tight_layout()
            plt.savefig(output_path + "seasonality_" + str(period_num) + "_" + column + ".png", dpi=600)
            plt.close()
        else:
            pass
    
    return None

def capacity_plot(input_df, output_path):
    """Plot the load profile curve according to the threashold

    Args:
        input_df (dataframe): contain the data to be plotted
        output_path (string): path to save the plot

    Returns:
        None
    """
    
    sns.set_theme(style="whitegrid")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    start_time = '2022-01-01 00:00:00'
    end_time = '2023-11-10 08:00:00'
    temp_df = input_df.drop(["Datetime"], axis=1)
    
    input_df = input_df.loc[(input_df['Datetime'] >= start_time) & (input_df['Datetime'] <= end_time)]
    color_list = ["#0466c8", "#780000", "#780000", 
        "#0466c8", "#d90429",  "#0353a4", "#023e7d", "#002855", "#001845", "#001233", "#33415c", "#5c677d", "#7d8597", "#979dac", ]

    column_names = temp_df.columns
    subplot_num = 2
    fig, axes = plt.subplots(subplot_num, 1, figsize=(20, 7))
    for iter in range(subplot_num):

        sns.lineplot(data=input_df, x='Datetime', y=column_names[iter], ax=axes[iter], color=color_list[iter])
        axes[iter].set_title(column_names[iter])

        n = 1000  # Set the desired frequency of ticks
        ticks = input_df.iloc[::n, 0]  # Select every nth tick from the 'Date' column
        axes[iter].set_xticks(ticks)
        axes[iter].set(xlabel="", ylabel="")

            
    # Hide xticks for all subplots except the bottom one
    for ax in axes[:-1]:
        ax.xaxis.set_tick_params(which='both', bottom=False, top=False, labelbottom=False)

    # Display xticks only at the bottom subplot
    axes[-1].xaxis.set_ticks_position('both')  # Display ticks on both sides
    axes[-1].xaxis.set_tick_params(which='both', bottom=True, top=False)  # Only bottom ticks are visible
    
    
    # Rotate the x-tick labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout
    plt.tight_layout()
    # Show the plot
    plt.savefig(output_path + "capacity_load_profile.png", dpi=600)
    plt.close()
    
    return None

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
    
    def corrfunc(x, y, **kwds):
        cmap = kwds['cmap']
        norm = kwds['norm']
        ax = plt.gca()
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        sns.despine(ax=ax, bottom=True, top=True, left=True, right=True)
        r, _ = pearsonr(x, y)
        facecolor = cmap(norm(r))
        ax.set_facecolor(facecolor)
        lightness = (max(facecolor[:3]) + min(facecolor[:3]) ) / 2
        ax.annotate(f"r={r:.2f}", xy=(.5, .5), xycoords=ax.transAxes,
                color='white' if lightness < 0.7 else 'black', size=26, ha='center', va='center')
    
    
    g = sns.PairGrid(input_df)
    g.map_lower(plt.scatter, s=10)
    g.map_diag(sns.histplot, kde=False)
    g.map_upper(corrfunc, cmap=plt.get_cmap('crest'), norm=plt.Normalize(vmin=-.5, vmax=.5))
        
    plt.tight_layout()
    plt.savefig(output_path + "/correlation_" + city_num + ".png", dpi=600)
    plt.close()

    return None

def extreme_weather_detect(input_df, output_path, start_date, end_date):
    """Detect the extreme weather based on conditions

    Args:
        input_df (dataframe): contain the weather data
        output_path (string): path to save the extreme weather data
        start_date (string): the start date of extreme weather
        end_date (string): the end date of extreme weather

    Returns:
        None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    time_index = pd.date_range(start=start_date, end=end_date, freq='H')
    # Create a DataFrame with the time series column
    datetime_df = pd.DataFrame({'Datetime': time_index})
        
    station_set = set(input_df["Closest_Station"])
    xlsx_base = "./result/NCDC_weather_data/stations_imputed/"
    
    for element in station_set:
        temp_xlsx_path = xlsx_base + str(element) + ".xlsx"
        temp_weather_df = pd.read_excel(temp_xlsx_path)
        temp_weather_df['DATE'] = pd.to_datetime(temp_weather_df['DATE'], format='%Y-%m-%d').dt.floor('D')
        
        city_df = input_df.loc[input_df['Closest_Station'] == element]
        city_num = next(iter(set(city_df["City"])))
        print("City", city_num)
        extreme_weather_df = datetime_df
        
        # Heat Index
        def calculate_heat_index(temp_celsius, relative_humidity):
            """Constants for the Heat Index calculation
            Input is celsius
            Coefficients are retrieved here
            https://en.wikipedia.org/wiki/Heat_index

            Args:
                temp_celsius (float): _description_
                relative_humidity (float): _description_

            Returns:
                float: 
            """
            
            c1 = -8.78469475556
            c2 = 1.61139411
            c3 = 2.33854883889
            c4 = -0.14611605
            c5 = -0.012308094
            c6 = -0.0164248277778
            c7 = 0.002211732
            c8 = 0.00072546
            c9 = -0.000003582

            # Calculate the Heat Index
            heat_index = (
                c1 +
                c2 * temp_celsius +
                c3 * relative_humidity +
                c4 * temp_celsius * relative_humidity +
                c5 * temp_celsius ** 2 +
                c6 * relative_humidity ** 2 +
                c7 * temp_celsius ** 2 * relative_humidity +
                c8 * temp_celsius * relative_humidity ** 2 +
                c9 * temp_celsius ** 2 * relative_humidity ** 2
            )

            # Convert Heat Index from Fahrenheit to Celsius
            heat_index_celsius = (heat_index - 32) * 5/9

            return heat_index_celsius
        
        temp_weather_df['Heat Index'] = calculate_heat_index(temp_weather_df['TEMP'], temp_weather_df['RH'])
        
        extreme_weather_df['Heat Index Caution'] = np.nan
        high_hum_df = temp_weather_df.loc[(temp_weather_df['Heat Index'] > 27) & 
                                               (temp_weather_df['Heat Index'] <= 32)]
        for date in high_hum_df['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Heat Index Caution'] = 1
                    
                    
        extreme_weather_df['Heat Index Extreme Caution'] = np.nan
        high_hum_df = temp_weather_df.loc[(temp_weather_df['Heat Index'] > 32) & 
                                               (temp_weather_df['Heat Index'] <= 41)]
        for date in high_hum_df['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Heat Index Extreme Caution'] = 1
        
        
        extreme_weather_df['Heat Index Danger'] = np.nan
        high_hum_df = temp_weather_df.loc[(temp_weather_df['Heat Index'] > 41) & 
                                               (temp_weather_df['Heat Index'] <= 54)]
        for date in high_hum_df['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Heat Index Danger'] = 1
        
        
        extreme_weather_df['Heat Index Extreme Danger'] = np.nan
        high_hum_df = temp_weather_df.loc[(temp_weather_df['Heat Index'] > 54)]
        for date in high_hum_df['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Heat Index Extreme Danger'] = 1
        print("Heat Index done")

        
        # Wind Chill
        def calculate_wind_chill_index(temp_celsius, wind_speed_mps):
            """# Calculate wind chill index

            Args:
                temp_celsius (float): the temperature in celcius
                wind_speed_mps (float): the wind speed in mile per second

            Returns:
                float: American wind chill index
            """
            
            wind_chill_index_us = (
                35.74 + 
                0.6215 * temp_celsius - 
                35.75 * wind_speed_mps ** 0.16 + 
                0.4275 * temp_celsius * wind_speed_mps ** 0.16
            )
            return wind_chill_index_us
        
        temp_weather_df['Wind Chill'] = calculate_wind_chill_index(temp_weather_df['TEMP'], temp_weather_df['WDSP'])
        
        extreme_weather_df['Wind Chill Very Cold'] = np.nan
        high_hum_df = temp_weather_df.loc[(temp_weather_df['Heat Index'] > -35) & 
                                               (temp_weather_df['Heat Index'] <= -25)]
        for date in high_hum_df['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Chill Very Cold'] = 1
        
        
        extreme_weather_df['Wind Chill Frostbite Danger'] = np.nan
        high_hum_df = temp_weather_df.loc[(temp_weather_df['Heat Index'] > -60) & 
                                               (temp_weather_df['Heat Index'] <= -35)]
        for date in high_hum_df['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Chill Frostbite Danger'] = 1
                    
        
        extreme_weather_df['Wind Chill Great Frostbite Danger'] = np.nan
        high_hum_df = temp_weather_df.loc[(temp_weather_df['Heat Index'] <= -60)]
        for date in high_hum_df['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Wind Chill Great Frostbite Danger'] = 1
        print("Wind Chill done")
        
        
        # Storm
        extreme_weather_df['Tropical Storm'] = np.nan
        thunderstorm_39_54 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > (39 * 0.44704)) & 
                                               (temp_weather_df['MXSPD'] <= (54 * 0.44704))]
        for date in thunderstorm_39_54['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Tropical Storm'] = 1
        print("Tropical Storm done")
        
        extreme_weather_df['Severe Tropical Storm'] = np.nan
        thunderstorm_54_73 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > (54 * 0.44704)) & 
                                               (temp_weather_df['MXSPD'] <= (73 * 0.44704))]
        for date in thunderstorm_54_73['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Severe Tropical Storm'] = 1
        print("Severe Tropical Storm done")
        
        extreme_weather_df['Typhoon'] = np.nan
        thunderstorm_73_93 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > (73 * 0.44704)) & 
                                               (temp_weather_df['MXSPD'] <= (93 * 0.44704))]
        for date in thunderstorm_73_93['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Typhoon'] = 1
        print("Typhoon done")
        
        extreme_weather_df['Strong Typhoon'] = np.nan
        thunderstorm_93_114 = temp_weather_df.loc[(temp_weather_df['MXSPD'] > (93 * 0.44704)) & 
                                               (temp_weather_df['MXSPD'] <= (114 * 0.44704))]
        for date in thunderstorm_93_114['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Strong Typhoon'] = 1
        print("Strong Typhoon done")
        
        extreme_weather_df['Super Typhoon'] = np.nan
        thunderstorm_114 = temp_weather_df.loc[temp_weather_df['MXSPD'] > (114 * 0.44704)]
        for date in thunderstorm_114['DATE']:
            # Iterate over the rows of the DataFrame
            for index, row in extreme_weather_df.iterrows():
                # Compare year, month, day, and hour of "Datetime" column with the target timestamp
                if row['Datetime'].year == date.year and \
                    row['Datetime'].month == date.month and \
                    row['Datetime'].day == date.day:
                    extreme_weather_df.at[index, 'Super Typhoon'] = 1
        print("Super Typhoon done")
        
        extreme_weather_df.to_excel(output_path + "extreme_weather_" + str(city_num) + ".xlsx", index=False)
    
    return None

def extreme_weather_plot(input_df, city, weather_data_path, start_time, end_time, output_path):
    """Plot the extreme weather

    Args:
        input_df (dataframe): contain the power data
        city (string): city to plot (all)
        weather_data_path (string): path to the extreme weather data
        start_time (string): the start date of extreme weather
        end_date (string): the end date of extreme weather
        output_path (string): path to save the plot

    Returns:
        None
    """
    sns.set_theme(style="whitegrid")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    weather_df = pd.read_excel(weather_data_path)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    time_index = pd.date_range(start=start_time, end=end_time, freq="h")
    # Create a DataFrame with the time series column
    time_series_df = pd.DataFrame({'Datetime': time_index})
    weather_df = pd.merge(time_series_df, weather_df, on='Datetime', how="left")
    # Filter out missing values from weather_df
    weather_df_filtered = weather_df.fillna("None")
    
    time_series_df = pd.merge(time_series_df, input_df, on='Datetime', how="left")
    time_series_df = pd.merge(time_series_df, weather_df_filtered, on='Datetime', how="left")
    time_series_df = time_series_df.set_index("Datetime")
    
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
    

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(time_series_df.index, time_series_df['Power'], color='#274c77')

    # Set background color according to Event values
    for event, color in event_colors.items():
        subset = time_series_df[time_series_df['Event'] == event]
        print(event)
        
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

        # Add the last part of the DataFrame
        if end_idx is not None:
            dfs.append(subset.loc[start_idx:end_idx])
        
        for group_df in dfs:
            if event == "None":
                ax.axvspan(group_df.index[0], group_df.index[-1], facecolor=color, alpha=0)
            else:
                ax.axvspan(group_df.index[0], group_df.index[-1], facecolor=color, alpha=0.6)
    
    
    
    legend = plt.legend()
    legend.get_frame().set_facecolor('none')
    plt.legend(frameon=False)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)        
    ax.set(xlabel="", ylabel="")
    
    plt.tight_layout()
    plt.savefig(output_path + "extreme_weather_" + city + ".png", dpi=600)
    plt.close()
    
    return None

def extreme_weather_city_plot(input_df, city, weather_data_path, start_time, end_time, output_path):
    """Plot extreme weather load profile for each city

    Args:
        input_df (dataframe): contain the power data
        city (string): city to plot
        weather_data_path (string): path to the extreme weather data
        start_time (string): the start date of extreme weather
        end_date (string): the end date of extreme weather
        output_path (string): path to save the plot

    Returns:
        None
    """
    sns.set_theme(style="whitegrid")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    weather_df = pd.read_excel(weather_data_path)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    time_index = pd.date_range(start=start_time, end=end_time, freq="h")
    # Create a DataFrame with the time series column
    time_series_df = pd.DataFrame({'Datetime': time_index})
    weather_df = pd.merge(time_series_df, weather_df, on='Datetime', how="left")
    # Filter out missing values from weather_df
    weather_df_filtered = weather_df.fillna("None")
    
    time_series_df = pd.merge(time_series_df, input_df, on='Datetime', how="left")
    time_series_df = pd.merge(time_series_df, weather_df_filtered, on='Datetime', how="left")
    time_series_df = time_series_df.set_index("Datetime")
    
    event_colors = {"Heat Index Caution":                   "#ad2831",
                    "Heat Index Extreme Caution":           "#800e13",
                    "Heat Index Danger":                    "#640d14",
                    "Heat Index Extreme Danger":            "#38040e",
                    
                    "Wind Chill Very Cold":                 "#0096c7",
                    "Wind Chill Frostbite Danger":          "#023e8a",
                    "Wind Chill Great Frostbite Danger":    "#03045e",
                    
                    "Tropical Storm":                       "#9d4edd",
                    "Severe Tropical Storm":                "#7b2cbf",
                    "Typhoon":                              "#5a189a",
                    "Strong Typhoon":                       "#3c096c",
                    "Super Typhoon":                        "#240046",}
    

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(time_series_df.index, time_series_df['Power'], color='#274c77')

    # Set background color according to Event values
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

            # Add the last part of the DataFrame
            if end_idx is not None:
                dfs.append(subset.loc[start_idx:end_idx])
        
            for group_df in dfs:
                ax.axvspan(group_df.index[0], group_df.index[-1], facecolor=color, alpha=0.6)
        
        
    legend = plt.legend()
    legend.get_frame().set_facecolor('none')
    plt.legend(frameon=False)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)        
    ax.set(xlabel="", ylabel="")
    
    plt.tight_layout()
    plt.savefig(output_path + "extreme_weather_" + city + ".png", dpi=600)
    plt.close()
    
    return None


def holiday_plot(input_df, city, weather_data_path, start_time, end_time, output_path):
    """Plot the extreme weather

    Args:
        input_df (dataframe): contain the power data
        city (string): city to plot (all)
        weather_data_path (string): path to the extreme weather data
        start_time (string): the start date of extreme weather
        end_date (string): the end date of extreme weather
        output_path (string): path to save the plot

    Returns:
        None
    """
    sns.set_theme(style="whitegrid")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    weather_df = pd.read_excel(weather_data_path)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    time_index = pd.date_range(start=start_time, end=end_time, freq="h")
    # Create a DataFrame with the time series column
    time_series_df = pd.DataFrame({'Datetime': time_index})
    weather_df = pd.merge(time_series_df, weather_df, on='Datetime', how="left")
    # Filter out missing values from weather_df
    weather_df_filtered = weather_df.fillna("None")
    
    time_series_df = pd.merge(time_series_df, input_df, on='Datetime', how="left")
    time_series_df = pd.merge(time_series_df, weather_df_filtered, on='Datetime', how="left")
    time_series_df = time_series_df.set_index("Datetime")
    
    event_colors = {'New year':                     '#641220', 
                    'Dragon Boat Festival':         '#3fa34d', 
                    'International Labor Day':      '#22333b',
                    'Mid-autumn festival':          '#fdc500',
                    'National Day':                 '#e09f3e',
                    'Qingming':                     '#8b949a',
                    'Spring Festival':              '#e01e37',
                    'None':                         "#FFFFFF"}
    

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(time_series_df.index, time_series_df['Power'], color='#274c77')

    # Set background color according to Event values
    for event, color in event_colors.items():
        subset = time_series_df[time_series_df['Holiday'] == event]
        print(event)
        
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

        # Add the last part of the DataFrame
        if end_idx is not None:
            dfs.append(subset.loc[start_idx:end_idx])
        
        for group_df in dfs:
            if event == "None":
                ax.axvspan(group_df.index[0], group_df.index[-1], facecolor=color, alpha=0, edgecolor='none')
            else:
                ax.axvspan(group_df.index[0], group_df.index[-1], facecolor=color, alpha=0.5, edgecolor='none')
    
    legend = plt.legend()
    legend.get_frame().set_facecolor('none')
    plt.legend(frameon=False)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)        
    ax.set(xlabel="", ylabel="")
    
    plt.tight_layout()
    plt.savefig(output_path + "holiday_" + city + ".png", dpi=600)
    plt.close()
    
    return None
