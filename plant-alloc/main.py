#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 09:38:56 2020.

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
pd.options.display.max_rows = 350
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
regions = consumption[['NUTS_ID', 'NUTS_NAME', 'Country', 'capita',
                       'BC per cap', 'Collecti_2', 'Color', 'BC_per_reg',
                       'geometry']].cx[2300000:7300000, 1060000:]

# Quick visual check
fig, ax = plt.subplots()
regions.plot(ax=ax)
plants.plot(color='red', ax=ax)


def euclid_dist(a, b):
    """Compute the Euclidean distance between points `a` and `b`."""
    x1, y1 = a[0][0], a[1][0]
    x2, y2 = b[0][0], b[1][0]
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def get_dists(plants, regions, plant_id='name', reg_id='NUTS_ID'):
    """Compute the distances between sorting plants and consumption regions.

    Parameters
    ----------
    plants : geopandas.GeoDataFrame
        Dataframe containing plant geometry.
    regions : geopandas.GeoDataFrame
        Dataframe containing region geometry.
    plant_id : string, optional
        Name of the column that uniquely identifies plants.
        The default is 'name'.
    region_id : string, optional
        Name of the column that uniquely identifies regions.
        The default is 'NUTS_ID'.

    Returns
    -------
    plant_dists : pandas.DataFrame
        Dataframe containing distances between plants and regions
    """
    # Initialize dataframe with region IDs as index
    plant_dists = pd.DataFrame(index=regions[reg_id])
    # Iterate over plants
    for _, plant in plants.iterrows():
        # Iterate over regions
        for _, region in regions.iterrows():
            # Compute and add distance between plant and region
            plant_dists.loc[region[reg_id], plant[plant_id]] = (
                euclid_dist(plant.geometry.xy, region.geometry.centroid.xy))
    return plant_dists


def allocate_recycling(plants, regions, plant_id='name', reg_id='NUTS_ID',
                       capacity_col='capacity i', consump_col='BC_per_reg'):
    """Allocate regional beverage carton streams to processing plants.

    Allocates on the basis of plant capacity and regional volume,
    always taking proximity into account.

    Parameters
    ----------
    plants : geopandas.GeoDataFrame
        Dataframe containing plant geometry.
    regions : geopandas.GeoDataFrame
        Dataframe containing region geometry.
    plant_id : string, optional
        Name of the column that uniquely identifies plants.
        The default is 'name'.
    region_id : string, optional
        Name of the column that uniquely identifies regions.
        The default is 'NUTS_ID'.
    capacity_col : string, optional
        Name of the column that stores processing plant capacity.
        The default is 'capacity i'.
    consump_col : string, optional
        Name of the column that stores regional beverage carton consumption.
        The default is 'BC_per_reg'.

    Returns
    -------
    output : geopandas.GeoDataFrame
        A copy of the `regions` input df with an added column that designates
        which plant a region is assigned to.
    """
    distances = get_dists(plants, regions, plant_id=plant_id,
                          reg_id=reg_id)
    output = regions.copy()
    output['plant_alloc'] = None
    for _, plant in plants.iterrows():
        capacity = int(plant[capacity_col])
        reg_dist = distances[[plant[plant_id]]].sort_values(plant[plant_id])
        # Cycle through plants from nearest to furthest
        for reg_idx, row in reg_dist.iterrows():
            consumption = np.float(
                regions[regions[reg_id] == reg_idx][consump_col])
            if consumption > capacity:
                break
            else:
                out_idx = output[output[reg_id] == reg_idx].index[0]
                capacity -= consumption
                output.loc[out_idx, 'plant_alloc'] = plant[plant_id]

    return output


test = allocate_recycling(plants, regions)

test.to_file("out/regions_w_plant_alloc.geojson", driver='GeoJSON')
