## Swiss data quality: augmenting CAMELS-CH with isotopes, water quality, agricultural and atmospheric chemistry data 
> April 2025

Authors: Thiago V. M. do Nascimento, Marvin Höge, Ursula Schönenberger, Sandra Pool, Rosi Siber, Martina Kauzlaric, Maria Staudinger, Pascal Horton, Marius G. Floriancic, Florian Storck, Päivi Rinta, Jan Seibert and Fabrizio Fenicia

Contact: thiago.nascimento@eawag.ch

### Folders and content
- **catchment_aggregated_data/**: contains `five` subfolders. Each is organized containing `one` csv-file per catchment, `115` files in total. The files are organized by time series (rows) and attribute variables (columns).
	- **agricultural_data/**: contains the crop-types data. 
		- camels_ch_chem_swisscrops{basin_id}.csv
	- **atmospheric_deposition/**: contains the atmospheric deposition data.
		- camels_ch_chem_atmdepo_{basin_id}.csv
	- **landcover_data/**: contains the landcover classes.
		- camels_ch_chem_landcover_{basin_id}.csv
	- **livestock_data/**: contains the livestock unit data.
		- camels_ch_chem_livestock_{basin_id}.csv
	- **rain_water_isotopes/**: contains the rain water catchment aggregated data. 
		- camels_ch_chem_rainisotopes_{basin_id}.csv

- **shapefiles:** Contains `three` subfolders. 
	- **camels_ch_del/**: contains two `shapefiles`. `One` shapefile includes the derived catchment boundaries associated with each gauge, and the `other` shapefile marks the location of these gauges. Both files are referenced in in the Swiss coordinate system LV95 (sometimes also referred to as CH1903+), and were copied from the original CAMELS-CH. 
		- camels_ch_chem_catchment_boundaries.{cpg,dbf,prj,shp,shx}
		- camels_ch_chem_gauging_stations.{cpg,dbf,prj,shp,shx}
	- **nawa_trend_del/**: provides the alternative catchment delineation shapefile for the NAWA TREND catchments. 
		- camels_ch_chem_catchment_boundaries_nawat.{cpg,dbf,prj,shp,shx}
	- **nawa_fracht_del/**: like the anterior, but for NAWA FRACHT catchments.
		- camels_ch_chem_catchment_boundaries_nawaf.{cpg,dbf,prj,shp,shx}

- **gauges_medatada/**: contains `one` csv-file covering all the metadata associated with each of the `115` gauging stations.
	- camels_ch_chem_gauges_metadata.csv

- **stream_water_chemistry/**: contains `two` subfolders.  
	- **timeseries/**: contains `two` nested sub-sub folders for respectively daily and hourly time steps. The csv-files in both are organized by time series (rows) and attribute variables (columns), and each column represents one of the water quality variable. Both nested subfolders contain `86` files. 
		- **daily/**:
			- camels_ch_chem_daily_{basin_id}.csv
		- **hourly/**:
			- camels_ch_chem_hourly_{basin_id}.csv
	- **interval_samples/**: contains `three` nested sub-subfolders. 
		- **nawa_fracht/**: contains one csv-file per catchment covered (`24` files). The rows represent the dates, and each column represents one of the water quality variables.
			- camels_ch_chem_nawafracht_{basin_id}.csv
		- **nawa_trend/**: contains one csv-file per catchment covered (`76 files`) presents a similar structure as the anterior, but for NAWA TREND data.
			- camels_ch_chem_nawatrend_{basin_id}.csv

- **stream_water_isotopes:** contains `two` subfolders. 
	- **isot/**: contains `nine` files in total and were derived from the ISOT dataset. 
		- camels_ch_chem_isot_{basin_id}.csv
	- **ch_irp/**: contains `11` files in total and were derived from the CH-IRP dataset. 
		- camels_ch_chem_chirp_{basin_id}.csv

## Usage notes

Due to potential measurement errors, the following variables from NAWA FRACHT should be used with caution:  
•	Mg between 1.1.1990 and 31.12.2009 for all stations.
•	pH_lab between 1.1.2000 and 1.1.2018 for all stations.
•	pH_sensor between 1.1.2003 and 1.1.2014 for the following stations: 
	o	Chancy, Aux Ripes - Rhône, nawaf_id 1838
	o	Diepoldsau - Rhein, nawaf_id 1821
	o	Hagneck - Aare, nawaf_id 1832
	o	Mellingen - Reuss, nawaf_id 1835
	o	Porte du Scex - Rhône, nawaf_id 1837
	o	Rekingen - Rhein, nawaf_id 1825
	o	Riazzino - Ticino, nawaf_id 1840
	o	S-chanf - Inn, nawaf_id 2064
	o	Weil, Palmrainbrücke - Rhein, nawaf_id 2078

## References
Currently CAMELS-CH-Chem covers `115` rivers catchments within Swiss territory. Please refer to the CAMELS-CH-Chem preprint, for a detailed description of the current dataset, including their respective variables units. For the codes used to derive the current dataset, users can refer to the [CAMELS-CH-Chem GitHub](https://github.com/camels-ch/camels-ch-chem). 

## How to cite
Medeiros do Nascimento, T. V., Schönenberger, U., Höge, M., Pool, S., Siber, R., Kauzlaric, M., Horton, P., Günter Floriancic, M., Staudinger, M., Storck, F., Rinta, P., Seibert, J., & Fenicia, F. (2025). Swiss data quality: augmenting CAMELS-CH with data on isotopes, water quality, agricultural data and atmospheric chemistry (0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14980027