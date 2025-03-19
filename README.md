# District power

## Introduction

Here contains the raw data and scripts for data processing and visualization for paper "[A Large-Scale Residential Load Dataset in a Southern Province of China](https://www.nature.com/articles/s41597-025-04766-7)". The data is deposited at [Figshare](https://doi.org/10.6084/m9.figshare.26333452.v1), here we include the analysis of the data for demonstration. The tree structure of this repository is shown below.

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
|        Weather Data        |                                                  [NOAA](https://www.ncei.noaa.gov/cdo-web/)                                                  |
|          Holiday          |                                       Collected from the Internet                                       |
|  Extreme Weather Internet  | [Collected from the media](http://news.gxnews.com.cn/staticpages/20240110/newgx659e5917-21404408.shtml#/) |
| Extreme Weather Calculated |                                    Calculated from the weather data                                    |
|      Transformer Data      |                                               Power Grid                                               |

## Usage Note

To run the code, you need to first download the code and data from figshare, move the figshare data into folder "data" (or in this case just download all Github files), install the dependencies in "requirements.txt", then run script "main.py".

This repository is under MIT License, please feel free to use. If you find this repository helpful, please cite the following bibtex entry:

```
@Article{Li2025,
    author={Li, Bo
        and Yu, Ruotao
        and Gan, Kaiye
        and Ruan, Guangchun
        and Liu, Shangwei
        and Yang, Mingxia
        and Xie, Daiyu
        and Dai, Wei
        and Zhong, Haiwang},
    title={A Large-Scale Residential Load Dataset in a Southern Province of China},
    journal={Scientific Data},
    year={2025},
    month={Mar},
    day={18},
    volume={12},
    number={1},
    pages={450},
    abstract={Granular, localized data are essential for generating actionable insights that facilitate the transition to a net-zero energy system, particularly in underdeveloped regions. Understanding residential electricity consumption---especially in response to extreme weather events such as heatwaves and tropical storms---is critical for enhancing grid resilience and optimizing energy management strategies. However, such data are often scarce. This study introduces a comprehensive dataset comprising hourly transformer-level residential electricity load data collected between 2022 and 2023 from 23 residential communities across 10 cities in Guangxi Province, China. The dataset is augmented with meteorological data, including temperature, humidity, and records of extreme weather events. Additionally, calendar-related data (e.g., holidays) are included to facilitate the analysis of consumption patterns. The paper provides a detailed overview of the methodologies employed for data collection, preprocessing, and analysis, with a particular emphasis on how extreme weather influences electricity demand in residential areas. This dataset is anticipated to support future research on energy consumption, climate change adaptation, and grid resilience.},
    issn={2052-4463},
    doi={10.1038/s41597-025-04766-7},
    url={https://doi.org/10.1038/s41597-025-04766-7}
}
```

## Contact

For questions or comments, you can reach me at [yuruotao@outlook.com](yuruotao@outlook.com).
