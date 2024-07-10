# District power

## Introduction

Here contains the raw data and scripts for data processing and visualization for paper "[District Power]()". The data is deposited at [Figshare](), here we include the analysis of the data for demonstration. The tree structure of this repository is shown below.

```yaml
    District_Power_Dataset
    ├── data				# Contain the data to be analyzed (download from figshare)
    │    ├── extreme_weather.xlsx	# Extreme weather event collected from the Internet
    │    ├── festival.xlsx		# Official festival date
    │    ├── isd-history.xlsx		# Weather station information
    │    ├── meta.xlsx			# The nearest weather station of each transformer
    │    ├── raw_data.xlsx		# Raw transformer data
    │    └── raw_data_adjusted.xlsx	# Transformer data with disqualified data deletion
    ├── result
    │    ├── aggregate			# Contain transformer data aggregated by district, city, and province
    │    ├── basic_statistics		# Calculate the numerical feature of the data
    │    ├── capacity			# Load curve for high capacity transformers and low capacity transformers
    │    ├── diversity_factor		# Diversity factor by day and hour for transformers
    │    ├── extreme_weather		# Extreme weather related results and plots
    │    ├── festival			# Plot the load curve with respect to festivals
    │    ├── imputation			# Store the result of imputation
    │    ├── load_profile		# Plot the load profile for all levels
    │    ├── missing_value		# Plot the distribution of missing value for each transformer
    │    ├── NCDC_weather_data		# NCDC weather data for each station in 2022 and 2023
    │    ├── profile			# Data profiling
    │    ├── resample			# Load curve with different sample rates
    │    ├── seasonality		# Explore the seasonality of load curve
    │    ├── select			# Plot load curve for specific transformer at specific time
    │    └── uniform			# Contain the uniformed data 
    ├── script  
    │    ├── analysis.py		# Analyze the factors such as weather and holidays regarding load curve
    │    ├── basic_statistics.py	# Calculate the basic statistics
    │    ├── data_obtain.py		# Obtain the NCDC weather data
    │    ├── imputation.py		# Perform imputation on missing value
    │    ├── missing_value.py		# Plot the distribution of missing value for transformers
    │    ├── profiling.py		# Profiling of data
    │    ├── resample.py		# Sample the data at different frequencies
    │    └── uniform.py			# Unify the data by dividing the rated capacity
    ├── main.py				# Analysis flow of the dataset
    ├── select_district.py		# Plot the load curve for a specific district
    ├── LICENSE
    ├── requirements.txt		# Dependencies for the project
    └── README.md
```

The sources of data are summarized here.

|            Data            |                                               Data Source                                               |
| :------------------------: | :-----------------------------------------------------------------------------------------------------: |
| Weather Stations Meta Data |                  [NOAA](https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/)                  |
|        Weather Data        |                                                  [NOAA]()                                                  |
|          Festival          |                                       Collected from the Internet                                       |
|      Extreme Weather      | [Collected from the media](http://news.gxnews.com.cn/staticpages/20240110/newgx659e5917-21404408.shtml#/) |
|      Transformer Data      |                                        China Southern Power Grid                                        |

## Usage Note

To run the code, you need to first download the code and data from figshare, move the figshare data into folder "data" (or in this case just download all Github files), install the dependencies in "requirements.txt", then run script "main.py".

This repository is under MIT License, please feel free to use. If you find this repository helpful, please cite the following bibtex entry:

```
@article{,
  title={},
  author={yuruotao},
  year={2024}
}
```

## Contact

For questions or comments, you can reach me at [yuruotao@outlook.com](yuruotao@outlook.com).
