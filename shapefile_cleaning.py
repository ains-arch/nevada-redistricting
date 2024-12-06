from functools import partial
import warnings
import argparse
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from gerrychain import Graph, Partition, GeographicPartition, MarkovChain, constraints
from gerrychain.tree import recursive_tree_part
from gerrychain.updaters import cut_edges, Tally
from gerrychain.proposals import recom
from gerrychain.accept import always_accept
import maup
from shapely.geometry import Polygon
from maup import smart_repair, quick_repair, resolve_overlaps
from geopandas import GeoSeries, GeoDataFrame

import warnings

# Suppress only FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning) # holy shit shut up about chainedassignmenterror

# Load the shapefile
original_gdf = gpd.read_file("data/shapefiles/precinct_p.shp")

# Information
print("info:\n")
original_gdf.info(verbose=True)

print("\nmaup doctor on unrepaired\n")
maup.doctor(original_gdf)
print("\nend maup doctor on unrepaired\n")

# Plot original gdf
original_gdf.plot()
plt.savefig("figs/original_gdf.png")
print("Saved figs/original_gdf.png\n")

# Smart repair
print("running smart repair on original gdf\n")
repaired_gdf = smart_repair(original_gdf)
print("\nend smart repair on original gdf\n")

# Check that repair succeeded: 
print("maup doctor on repaired gdf")
maup.doctor(repaired_gdf)
print("end maup doctor on repaired gdf\n")

# Dual graph sanity check
print("loading repaired gdf graph")
# graph = Graph.from_file(PATH)
graph = Graph.from_geodataframe(repaired_gdf)
print("repaired gdf graph loaded\n")

# Information about graph
print(f"Is the dual graph connected? {nx.is_connected(graph)}")
print(f"Is the dual graph planar? {nx.is_planar(graph)}")
print(f"Number of Nodes: {len(graph.nodes())}")
print(f"Number of Edges: {len(graph.edges())}\n")

# Plot repair one
repaired_gdf.plot()
plt.savefig("figs/repair_1.png")
print("Saved figs/repaired.png\n")

# Check geometries

# # Save invalid geometries
# invalid_geom_indexes = ~gdf['geometry'].is_valid
# invalid_geom = gdf[invalid_geom_indexes]

# # Fix invalid geometries
# gdf['geometry'] = gdf['geometry'].buffer(0)

# for id in invalid_geom.GlobalID:
    # # plot old geometry
    # invalid_geom[invalid_geom['GlobalID'] == id].plot()
    # # plot new geometry
    # gdf[gdf['GlobalID'] == id].plot()

# plt.savefig("figs/invalid.png")

# # Verify if there are still invalid geometries
# print("Valid count:", gdf.is_valid.value_counts())

# Check for remaining invalid geometries
# if not gdf.is_valid.all()
    # print("Still contains invalid geometries")
# else:
    # print("All geometries are valid")

# Print datatypes

# Drop the Shape_Area column, we don't need area in this analysis
print("getting rid of area\n")
no_area_gdf = repaired_gdf.drop(columns=['Shape_Area'])

# Save the repaired shapefile without the Shape_Area column
# gdf.to_file("data/precinct_p_fixed.shp")

# Set path to shapefile
# PATH = "data/precinct_p_fixed.shp"
# print(f"Shapefile path: {PATH}\n")

# Geodataframe from shapefile
# print("Loading Geodataframe...")
# gdf = gpd.read_file(PATH)
# print("Geodataframe loaded.\n")

# Check the geometry column
print("Geometry type:", no_area_gdf['geometry'].geom_type.unique())

# Check if 'PREC' is unique
print("getting rid of duplicate PREC")
is_unique = no_area_gdf['PREC'].is_unique
print("Is 'PREC' unique?:", is_unique)

# Identify rows with duplicate PREC values
duplicates = no_area_gdf[no_area_gdf.duplicated(subset='PREC', keep=False)]
print("Duplicate PREC rows:\n", duplicates)

# Check surrounding PREC values
# surrounding = gdf[gdf['PREC'].isin([7900,7901,7902,7903,7904,7905,7906,7907,7908,7909,7910])]
# print("neighborhood:", surrounding)

# Check if GlobalID aligns with 'PREC'
# print("Is 'GlobalID' unique?:", no_area_gdf['GlobalID'].is_unique)

# Check for invalid geometries
# invalid_geom = no_area_gdf[~no_area_gdf.geometry.is_valid]
# print("Number of invalid geometries:", len(invalid_geom))

# Filter the two rows with PREC value 7908
precinct_7908 = no_area_gdf[no_area_gdf['PREC'] == 7908]

# Merge the geometries of the precinct (using unary_union)
print("union weird precinct")
merged_geometry = precinct_7908.geometry.unary_union
print("done unioning weird precinct")

# Update the GeoDataFrame with the merged geometry (keep other attributes intact)
merged_gdf = no_area_gdf.copy()
print("merge unioned weird precinct with gdf")
merged_gdf.loc[merged_gdf['PREC'] == 7908, 'geometry'] = merged_geometry

# Plot the updated GeoDataFrame
plt.figure()
merged_gdf.plot(edgecolor='black', linewidth=0.2)
plt.axis('off')
plt.savefig("figs/merged.png")
print("\nSaved figs/merged.png")

# # Separate them into two precincts
# precinct_7908_first = precinct_7908.iloc[0:1]
# precinct_7908_second = precinct_7908.iloc[1:2]

# # Plotting
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# # Plot the first precinct with one color (e.g., red)
# precinct_7908_first.plot(ax=ax, color='red', alpha=0.7, label="7908 (First)")

# # Plot the second precinct with another color (e.g., blue)
# precinct_7908_second.plot(ax=ax, color='blue', alpha=0.7, label="7908 (Second)")

# # Add labels for each precinct
# for idx, row in precinct_7908_first.iterrows():
    # centroid = row['geometry'].centroid
    # ax.text(centroid.x, centroid.y, str(row['PREC']), fontsize=8, ha='center', color='black')

# for idx, row in precinct_7908_second.iterrows():
    # centroid = row['geometry'].centroid
    # ax.text(centroid.x, centroid.y, str(row['PREC']), fontsize=8, ha='center', color='black')

# # Title and legend
# plt.title("Precincts 7908 (Two Different Colors)")
# plt.savefig("figs/dubious_precincts.png")

# Remove the problematic row with PREC == 7908 (the weird line)
# gdf_cleaned = gdf[gdf['PREC'] != 7908].copy()

# Use maup to ensure all geometries are valid
print("\nmaup doctor on merged gdf")
maup.doctor(merged_gdf)
print("end maup doctor on merged gdf")

# Try smart repair again
print("\nresolving overlaps")
no_overlap_geoseries = resolve_overlaps(merged_gdf, relative_threshold=None)
print("overlaps resolved")

# Check if repaired_gdf_cleaned is a GeoSeries
if isinstance(no_overlap_geoseries, gpd.GeoSeries):
    # Convert GeoSeries to GeoDataFrame by creating a new GeoDataFrame
    print("converting back to gdf")
    no_overlap_gdf = gpd.GeoDataFrame(geometry=no_overlap_geoseries)

# Check that repair succeeded: 
print("\nmaup doctor on resolved overlaps")
maup.doctor(no_overlap_gdf)
print("end maup doctor on resolved overlaps")

no_overlap_gdf.plot()
plt.savefig("figs/no_overlaps.png")

# Check dual graph
print("\ngraph sanity check no overlaps")
graph = Graph.from_geodataframe(no_overlap_gdf)

# Check geometry overlaps for specific indices
# from shapely.geometry import Polygon

# geom1 = duplicates.iloc[0]['geometry']
# geom2 = duplicates.iloc[1]['geometry']
# overlap = geom1.intersection(geom2)
# print(f"Overlap area: {overlap.area}")

# Plotting overlap (precincts 1003 and 1004)
# overlap_geom1 = gdf.iloc[1003]['geometry']
# overlap_geom2 = gdf.iloc[1004]['geometry']

# Plot the precincts 1003 and 1004
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# gdf.iloc[[1003, 1004]].plot(ax=ax, color='red', alpha=0.5)

# Plot the overlap if it exists
# if overlap_geom1.intersects(overlap_geom2):
    # overlap = overlap_geom1.intersection(overlap_geom2)
    # gpd.GeoSeries(overlap).plot(ax=ax, color='blue', alpha=0.5)

# plt.title("Overlap between precincts 1003 and 1004")
# plt.savefig("figs/overlap.png")
