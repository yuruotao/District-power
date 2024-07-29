# District power

## Introduction

Here contains the raw data and scripts for data processing and visualization for paper "[District Power]()". The data is deposited at [Figshare](), here we include the analysis of the data for demonstration. The tree structure of this repository is shown below.

```yaml
    District_Power_Dataset
    ├── data					# Contain the data to be analyzed
    │    ├── extreme_weather_internet.xlsx	# Extreme weather event collected from the Internet (Not provided)
    │    ├── holiday.xlsx			# Official holiday date
    │    ├── isd-history.xlsx			# Weather station meta information
    │    ├── transformer_meta.xlsx		# Meta data for transformers
    │    ├── transformer_raw.xlsx		# Raw transformer data
    │    └── Transformer_DB
    │    	└── Transformer_DB.db		# The aggregated database for analysis (download from figshare)
    │    └── guangxi_administration
    │    	├── guangxi.dbf
    │    	├── guangxi.prj
    │    	├── guangxi.shx
    │    	└── guangxi.shp			# Shapefile for Guangxi Province (download from figshare)
    ├── result
    ├── script  
    │    ├── database_create.py			# Script for creating database
    │    ├── diversity_factor.py		# Calculate the diversity factor and visualize
    │    ├── load_missing.py			# Handle the missing values in transformer data
    │    ├── load_profile.py			# Visualize the load profile at different scales
    │    └── weather.py				# Analyze the transformer data with respect to weather
    ├── main.py					# Analysis flow of the dataset
    ├── LICENSE
    ├── requirements.txt			# Dependencies for the project
    └── README.md
```

The sources of data are summarized here.

|            Data            |                                               Data Source                                               |
| :------------------------: | :-----------------------------------------------------------------------------------------------------: |
| Weather Stations Meta Data |                  [NOAA](https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/)                  |
|        Weather Data        |                                                  [NOAA]()                                                  |
|          Holiday          |                                       Collected from the Internet                                       |
|  Extreme Weather Internet  | [Collected from the media](http://news.gxnews.com.cn/staticpages/20240110/newgx659e5917-21404408.shtml#/) |
| Extreme Weather Calculated |                                    Calculated from the weather data                                    |
|      Transformer Data      |                                               Power Grid                                               |

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
