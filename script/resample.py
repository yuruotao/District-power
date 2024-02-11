import pandas as pd
import os



def resample(input_df, output_path, freq_list):
    
    print("Resample begin")
    datetime_column = input_df["Datetime"]
    temp_df = input_df.set_index(["Datetime"])
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    resample_list = []
    for freq in freq_list:
        aggregated_df = temp_df.resample(freq).sum()
        aggregated_df = aggregated_df.reset_index()
        aggregated_df = aggregated_df.drop(columns=["index"])
        
        aggregated_df.to_excel(output_path + "/resampled_" + freq + ".xlsx", index=False)
        resample_list.append(aggregated_df)
    
    return resample_list

def resample_visualization(input_df, resample_df_list):
    
    
    
    
    return None