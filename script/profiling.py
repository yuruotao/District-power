import pandas as pd
import os
from ydata_profiling import ProfileReport


def profiling(input_df, save_path):
    """create profiling for raw data

    Args:
        input_df (dataframe): _description_
        save_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    datetime_column = input_df["Datetime"]
    input_df = input_df.drop(columns=["Datetime"])
    column_list = input_df.columns.values.tolist()

    profile_dir = save_path + "/"
    
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)
    
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
        temp_df = pd.concat([datetime_column, temp_df], axis=1)
        index = list[0].split("-")
        
        profile = ProfileReport(
                temp_df,
                tsmode=True,
                sortby="Datetime",
                title="Time-Series EDA for City " + index[0] + " District " + index[1],
        )
        profile.to_file(profile_dir + "City-" + index[0] + "-" + "District-" + index[1] + ".html")
    
    
    return None