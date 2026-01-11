# PyMarineSurvey QC Report

- Metrics: `outputs\qc\qc_metrics.csv`
- Quicklooks: `outputs\quicklooks`

## Summary

- GREEN: 11
- YELLOW: 0
- RED: 0

## Details (grouped by QC score)

### RED

- (none)

### YELLOW

- (none)

### GREEN

#### multibeam_bathymetry | bathymetry_grid
- File: `bathymetry/calculated_positive_values/BSH_N-03-07_ETRS89_LAT_T20_Norbit_50cm_pos_num.tiff`

![quicklook](outputs/quicklooks/multibeam_bathymetry_bathymetry_grid_BSH_N-03-07_ETRS89_LAT_T20_Norbit_50cm_pos_num.png)

![hist](outputs/quicklooks/multibeam_bathymetry_bathymetry_grid_BSH_N-03-07_ETRS89_LAT_T20_Norbit_50cm_pos_num__hist.png)

- driver=GTiff | nodata=-3.40282e+38 | nodata_ratio=0.643575 | p01=29.4377 | p50=31.1957 | p99=33.3767 | crs=EPSG:25832 | width=14743 | height=17666 | pixel_size_x=0.5 | qc_notes=ok

#### side_scan_sonar | sss_mosaic
- File: `SSS/BSH_N-03-07_nodata251.tif`

![quicklook](outputs/quicklooks/side_scan_sonar_sss_mosaic_BSH_N-03-07_nodata251.png)

![hist](outputs/quicklooks/side_scan_sonar_sss_mosaic_BSH_N-03-07_nodata251__hist.png)

- driver=GTiff | nodata=251 | nodata_ratio=0.571226 | p01=88 | p50=156 | p99=212 | crs=EPSG:32632 | width=27716 | height=32115 | pixel_size_x=0.25 | qc_notes=ok

#### sub_bottom_horizon | horizon_depth
- File: `picked_horizons_tops/02_Top_Unit_Ib - Depth_Data.tif`

![quicklook](outputs/quicklooks/sub_bottom_horizon_horizon_depth_02_Top_Unit_Ib_-_Depth_Data.png)

![hist](outputs/quicklooks/sub_bottom_horizon_horizon_depth_02_Top_Unit_Ib_-_Depth_Data__hist.png)

- driver=GTiff | nodata=-1e+35 | nodata_ratio=0.507143 | p01=30.1281 | p50=31.8436 | p99=35.4931 | crs=PROJCS["ETRS89_UTM_zone_32N",GEOGCS["ETRF89",DATUM["European_Terrestrial_Reference_Frame_1989",SPHEROID["GRS 1980",6378137,298.257222101],AUTHORITY["EPSG","1178"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]] | width=564 | height=773 | pixel_size_x=10 | qc_notes=ok_sparse_layer

#### sub_bottom_horizon | horizon_depth
- File: `picked_horizons_tops/03_Top_Unit_II - Depth_Data.tif`

![quicklook](outputs/quicklooks/sub_bottom_horizon_horizon_depth_03_Top_Unit_II_-_Depth_Data.png)

![hist](outputs/quicklooks/sub_bottom_horizon_horizon_depth_03_Top_Unit_II_-_Depth_Data__hist.png)

- driver=GTiff | nodata=-1e+35 | nodata_ratio=0.507283 | p01=36.871 | p50=43.7654 | p99=49.5716 | crs=PROJCS["ETRS89_UTM_zone_32N",GEOGCS["ETRF89",DATUM["European_Terrestrial_Reference_Frame_1989",SPHEROID["GRS 1980",6378137,298.257222101],AUTHORITY["EPSG","1178"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]] | width=564 | height=773 | pixel_size_x=10 | qc_notes=ok_sparse_layer

#### sub_bottom_horizon | horizon_depth
- File: `picked_horizons_tops/04_Top_Unit_III - Depth_Data_rev1.tif`

![quicklook](outputs/quicklooks/sub_bottom_horizon_horizon_depth_04_Top_Unit_III_-_Depth_Data_rev1.png)

![hist](outputs/quicklooks/sub_bottom_horizon_horizon_depth_04_Top_Unit_III_-_Depth_Data_rev1__hist.png)

- driver=GTiff | nodata=-1e+35 | nodata_ratio=0.506497 | p01=42.1474 | p50=51.1387 | p99=81.1453 | crs=PROJCS["ETRS89_UTM_zone_32N",GEOGCS["ETRF89",DATUM["European_Terrestrial_Reference_Frame_1989",SPHEROID["GRS 1980",6378137,298.257222101],AUTHORITY["EPSG","1178"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]] | width=565 | height=774 | pixel_size_x=10 | qc_notes=ok_sparse_layer

#### sub_bottom_horizon | horizon_depth
- File: `picked_horizons_tops/05_Top_Unit_V_conservative - Depth_Data_rev1.tif`

![quicklook](outputs/quicklooks/sub_bottom_horizon_horizon_depth_05_Top_Unit_V_conservative_-_Depth_Data_rev1.png)

![hist](outputs/quicklooks/sub_bottom_horizon_horizon_depth_05_Top_Unit_V_conservative_-_Depth_Data_rev1__hist.png)

- driver=GTiff | nodata=-1e+35 | nodata_ratio=0.415992 | p01=77.2534 | p50=87.445 | p99=92.8334 | crs=PROJCS["ETRS89_UTM_zone_32N",GEOGCS["ETRF89",DATUM["European_Terrestrial_Reference_Frame_1989",SPHEROID["GRS 1980",6378137,298.257222101],AUTHORITY["EPSG","1178"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]] | width=311 | height=383 | pixel_size_x=10 | qc_notes=ok_sparse_layer

#### sub_bottom_horizon | horizon_depth
- File: `picked_horizons_tops/05_Top_Unit_V_larger - Depth_Data_rev1.tif`

![quicklook](outputs/quicklooks/sub_bottom_horizon_horizon_depth_05_Top_Unit_V_larger_-_Depth_Data_rev1.png)

![hist](outputs/quicklooks/sub_bottom_horizon_horizon_depth_05_Top_Unit_V_larger_-_Depth_Data_rev1__hist.png)

- driver=GTiff | nodata=-1e+35 | nodata_ratio=0.505515 | p01=77.9307 | p50=89.9028 | p99=100.277 | crs=PROJCS["ETRS89_UTM_zone_32N",GEOGCS["ETRF89",DATUM["European_Terrestrial_Reference_Frame_1989",SPHEROID["GRS 1980",6378137,298.257222101],AUTHORITY["EPSG","1178"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]] | width=565 | height=773 | pixel_size_x=10 | qc_notes=ok_sparse_layer

#### sub_bottom_thickness | sediment_thickness
- File: `sediment_thickness/60_Thickness_Unit_III_Data_rev1.tif`

![quicklook](outputs/quicklooks/sub_bottom_thickness_sediment_thickness_60_Thickness_Unit_III_Data_rev1.png)

![hist](outputs/quicklooks/sub_bottom_thickness_sediment_thickness_60_Thickness_Unit_III_Data_rev1__hist.png)

- driver=GTiff | nodata=-1e+35 | nodata_ratio=0.507419 | p01=7.27288 | p50=38.4251 | p99=52.8577 | crs=PROJCS["ETRS89_UTM_zone_32N",GEOGCS["ETRF89",DATUM["European_Terrestrial_Reference_Frame_1989",SPHEROID["GRS 1980",6378137,298.257222101],AUTHORITY["EPSG","1178"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]] | width=565 | height=772 | pixel_size_x=10 | qc_notes=ok_sparse_layer

#### sub_bottom_thickness | sediment_thickness
- File: `sediment_thickness/60_Thickness_Unit_II_Data_rev1.tif`

![quicklook](outputs/quicklooks/sub_bottom_thickness_sediment_thickness_60_Thickness_Unit_II_Data_rev1.png)

![hist](outputs/quicklooks/sub_bottom_thickness_sediment_thickness_60_Thickness_Unit_II_Data_rev1__hist.png)

- driver=GTiff | nodata=-1e+35 | nodata_ratio=0.507283 | p01=0.269353 | p50=7.57896 | p99=37.33 | crs=PROJCS["ETRS89_UTM_zone_32N",GEOGCS["ETRF89",DATUM["European_Terrestrial_Reference_Frame_1989",SPHEROID["GRS 1980",6378137,298.257222101],AUTHORITY["EPSG","1178"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]] | width=564 | height=773 | pixel_size_x=10 | qc_notes=ok_sparse_layer

#### sub_bottom_thickness | sediment_thickness
- File: `sediment_thickness/60_Thickness_Unit_Ia_Data_Data.tif`

![quicklook](outputs/quicklooks/sub_bottom_thickness_sediment_thickness_60_Thickness_Unit_Ia_Data_Data.png)

![hist](outputs/quicklooks/sub_bottom_thickness_sediment_thickness_60_Thickness_Unit_Ia_Data_Data__hist.png)

- driver=GTiff | nodata=-1e+35 | nodata_ratio=0.507306 | p01=0.305202 | p50=0.594871 | p99=2.14231 | crs=PROJCS["ETRS89_UTM_zone_32N",GEOGCS["ETRF89",DATUM["European_Terrestrial_Reference_Frame_1989",SPHEROID["GRS 1980",6378137,298.257222101],AUTHORITY["EPSG","1178"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]] | width=564 | height=773 | pixel_size_x=10 | qc_notes=ok_sparse_layer

#### sub_bottom_thickness | sediment_thickness
- File: `sediment_thickness/60_Thickness_Unit_Ib_Data.tif`

![quicklook](outputs/quicklooks/sub_bottom_thickness_sediment_thickness_60_Thickness_Unit_Ib_Data.png)

![hist](outputs/quicklooks/sub_bottom_thickness_sediment_thickness_60_Thickness_Unit_Ib_Data__hist.png)

- driver=GTiff | nodata=-1e+35 | nodata_ratio=0.506507 | p01=5.93663 | p50=11.5415 | p99=17.1258 | crs=PROJCS["ETRS89_UTM_zone_32N",GEOGCS["ETRF89",DATUM["European_Terrestrial_Reference_Frame_1989",SPHEROID["GRS 1980",6378137,298.257222101],AUTHORITY["EPSG","1178"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]] | width=564 | height=772 | pixel_size_x=10 | qc_notes=ok_sparse_layer
