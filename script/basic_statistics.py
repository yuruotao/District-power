import pandas as pd
import os
import numpy as np

def basic_statistics(input_df, output_path):


filepath = 'Electricity Consumption/'
dirs = os.listdir(filepath)
dirs.sort()
dirs = dirs[1:]

    STA = pd.DataFrame(index = dirs, 
                   columns = ['Mean',
                              'Standard deviation',
                              'Skew','Kurtosis',
                              '0th percentile',
                              '2.5th percentile',
                              '50th percentile',
                              '97.5th percentile',
                              '100th percentile'],
                   dtype = 'float')


    STA.loc[file,'Mean'] = data['Value'].mean(axis=0)
    STA.loc[file,'Standard deviation'] = data['Value'].std(axis=0)
    STA.loc[file,'Skew'] = data['Value'].skew(axis=0)
    STA.loc[file,'Kurtosis'] = data['Value'].kurtosis(axis=0)
    STA.loc[file,'0th percentile'] = data['Value'].quantile(q=0)
    STA.loc[file,'2.5th percentile'] = data['Value'].quantile(q=0.025)
    STA.loc[file,'50th percentile'] = data['Value'].quantile(q=0.5)
    STA.loc[file,'97.5th percentile'] = data['Value'].quantile(q=0.975)
    STA.loc[file,'100th percentile'] = data['Value'].quantile(q=1)#np.round(,2)
    STA = STA.round(2)
    STA.to_csv('/basic_statistics.csv', float_format='%.2f')