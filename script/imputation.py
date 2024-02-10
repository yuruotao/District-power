import pandas as pd
import os
from datetime import timedelta, datetime
from fancyimpute import IterativeImputer



def imputation(input_df, imputation_method, save_path):
    datetime_column = input_df["Datetime"]
    input_df = input_df.drop(columns=["Datetime"])
    
    imputation_dir = save_path + "/"
    
    if not os.path.exists(imputation_dir):
        os.makedirs(imputation_dir)

    if imputation_method == "MICE":
        imputer = IterativeImputer()
        imputed_data = imputer.fit_transform(input_df)
        print(imputed_data)
        # Convert the result back to a DataFrame
        imputed_df = pd.DataFrame(imputed_data, columns=input_df.columns, index=input_df.index)
    
    print(imputed_df)
    imputed_df = pd.concat([datetime_column, imputed_df], axis=1)
    imputed_df.to_excel(imputation_dir + "/" + "imputed_data_" + imputation_method + ".xlsx")
    
    return imputed_df

