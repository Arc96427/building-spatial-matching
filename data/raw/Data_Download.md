# Data Download

## OSM data
Please use Overpass to download and obtain it. The execution statement is as follows：

[out:json][timeout:25]; // Define the area (Munich) area[“name”=“München”][“boundary”=“administrative”]->.searchArea; // Search for buildings in the area (way[“building”](area.searchArea); relation[“building”](area.searchArea); ); out body; >; out skel qt;

## Official Data
See https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=hausumringe
Please select a fixed area to download.

# After downloading the original data, pay attention to the consistency of the EPSG. It is best to pre-check the consistency of the data specifications in GIS software.
