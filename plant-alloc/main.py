#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 09:38:56 2020

@author: peter
"""
# Import standard library
import pathlib

# Import libraries
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

# Import custom libraries

# Settings
pd.options.display.max_rows = 100
pd.options.display.max_columns = 50
plt.style.use('seaborn-colorblind')

# Set data directory
datadir = pathlib.Path('data')

# Load files
consumption = (gpd.read_file(datadir.joinpath('Regional_BC_2030.geojson'))
               .set_crs('EPSG:3035', allow_override=True))
plants = (gpd.read_file(datadir.joinpath('UBC_processing_2030.geojson'))
          .set_crs('EPSG:3035', allow_override=True))

# Keep only relevant columns and areas within Europe proper
consumption = consumption[['NUTS_ID', 'NUTS_NAME', 'Country', 'capita',
                           'BC per cap', 'Collecti_2', 'Color', 'BC_per_reg',
                           'geometry']].cx[2300000:7300000, 1060000:]

# Quick visual check
fig, ax = plt.subplots()
consumption.plot(ax=ax)
plants.plot(color='red', ax=ax)


def euclidean_dist(a, b):
    """Compute the Euclidean distance between two points."""
    x1, y1 = a[0][0], a[0][1]
    x2, y2 = b[0][0], b[0][1]

    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

for idx, plant in plants.iterrows():
    # Set max capacity
    plant_cap = int(plant['capacity i'])
    plant_loc = plant.geometry.xy
    for region in regions.iterrows():
        reg_consumption = region.BC_per_reg  # is this it??
        region_loc = region.geometry.centroid.xy
        # Compute distance
        





consumption.BC_per_reg




