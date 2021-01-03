#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:39:09 2021

@author: peterp
"""

# Import standard library
import pathlib

# Import libraries
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sciopt
from scipy.spatial import distance_matrix

# Set data directory
datadir = pathlib.Path('data')
outdir = pathlib.Path('out')

# Load files
consumption = (gpd.read_file(datadir.joinpath('Regional_BC_2030.geojson'))
               .set_crs('EPSG:3035', allow_override=True))
facs = (gpd.read_file(datadir.joinpath('UBC_processing_2030.geojson'))
        .set_crs('EPSG:3035', allow_override=True))

# Keep only relevant columns and areas within Europe proper
regs = consumption[['NUTS_ID', 'NUTS_NAME', 'Country', 'capita',
                    'BC per cap', 'Collecti_2', 'Color', 'BC_per_reg',
                    'geometry']].cx[2300000:7300000, 1060000:]

# Quick visual check
fig, ax = plt.subplots()
regs.plot(ax=ax)
facs.plot(color='red', ax=ax)

# Number of Facilities and Regions
N_F = len(facs)
N_R = len(regs)

# Facility capacity
C = np.array(facs['capacity i'].astype(np.float))
# Regional supply
S = np.array(regs.BC_per_reg)

# Coordinates of facilities and regions
F_coords = [[x, y] for x, y in zip(facs.geometry.x, facs.geometry.y)]
R_coords = [[x, y] for x, y in zip(regs.geometry.centroid.x,
                                   regs.geometry.centroid.y)]

# Distance matrix (regions = rows, facilities = cols) /1000 for km
dist = distance_matrix(R_coords, F_coords)/1000
# Distance vector
dist_vec = dist.flatten()

# Initialize basis vector of R^(len(facilities))
ones_N_F = np.ones(N_F)
# Define constraints
A_capacity = np.zeros((N_F, N_R * N_F))
for i in range(N_R):
    A_capacity[:, i * N_F:(i * N_F + N_F)] = np.diag(ones_N_F)

A_supply = np.zeros((N_R, N_R * N_F))
for i in range(N_R):
    A_supply[i, i * N_F:(i * N_F + N_F)] = ones_N_F

# Create solver
x_opt = sciopt.linprog(c=dist_vec, A_eq=A_capacity, b_eq=C,
                       A_ub=A_supply, b_ub=S, bounds=(0, None),
                       method='interior-point')

# Output to matrix
x_opt_mat = np.reshape(x_opt.x, (N_R, N_F))
# Normalize matrix for volume weighted plotting of supply lines
x_opt_mat_normalized = x_opt_mat/np.linalg.norm(x_opt_mat)

df = pd.DataFrame(x_opt_mat)
df.columns = facs.name
df.insert(loc=0, column='NUTS_region', value=regs.NUTS_ID)

fig, ax = plt.subplots(figsize=(15,15))
regs.plot(ax=ax)
facs.plot(color='red', ax=ax)

for i in range(N_R):
    ax.annotate(f'{S[i]-np.sum(x_opt_mat[i,:]):.1f}',
                (regs.iloc[i, :].geometry.centroid.x,
                 regs.iloc[i, :].geometry.centroid.y), fontsize=8)

    for j in range(N_F):
        # ax.annotate(C[j], (xvalues_facilities[j], yvalues_facilities[j]+0.5),fontsize=15)
        plt.plot([regs.iloc[i, :].geometry.centroid.x,
                  facs.iloc[j, :].geometry.x],
                 [regs.iloc[i, :].geometry.centroid.y,
                  facs.iloc[j, :].geometry.y], 'k-',
                 linewidth=x_opt_mat_normalized[i, j] * 10)
        if x_opt_mat_normalized[i, j] > 1e-8:
            ax.annotate("{:.1f}".format(x_opt_mat[i, j]),
                        ((regs.iloc[i, :].geometry.centroid.x +
                          facs.iloc[j, :].geometry.x)/2,
                         (regs.iloc[i, :].geometry.centroid.y +
                          facs.iloc[j, :].geometry.y)/2),
                        fontsize=15)
plt.savefig('test.svg')
plt.show()
