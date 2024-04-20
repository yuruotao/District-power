import pandas as pd
import os
import missingno as msno
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})

def missing_value_visualization(input_df, save_path):
    """visualize the missing data in both bar and matrix format

    Args:
        input_df (dataframe): the dataframe containing raw data
        save_path (string): folder to save the plot

    Returns:
        None
    """
    datetime_column = input_df["Datetime"]
    input_df = input_df.drop(columns=["Datetime"])
    column_list = input_df.columns.values.tolist()

    missing_dir_matrix = save_path + "/matrix/"
    missing_dir_bar = save_path + "/bar/"
    
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
        ax = msno.matrix(temp_df, fontsize=20, figsize=(16, 12), label_rotation=0)
        plt.xlabel("Transformers", fontsize=20)
        plt.ylabel("Sample Points", fontsize=20)
        plt.savefig(missing_dir_matrix + "City-" + index[0] + "-" + "District-" + index[1] +'-matrix.png', dpi=600)
        plt.close()
        
        # Bar plot
        ax = msno.bar(temp_df, fontsize=20, figsize=(16, 12), label_rotation=0)
        plt.xlabel("Transformers", fontsize=20)
        plt.ylabel("Sample Points", fontsize=20)
        plt.savefig(missing_dir_bar + "City-" + index[0] + "-" + "District-" + index[1] +'-bar.png', dpi=600)
        plt.close()
    
    return None