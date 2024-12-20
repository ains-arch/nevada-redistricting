import warnings
import re
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from gerrychain import Graph
import maup
from maup import smart_repair, resolve_overlaps
from shapely.ops import unary_union
from pyei.two_by_two import TwoByTwoEI
from pyei.goodmans_er import GoodmansER
from pyei.goodmans_er import GoodmansERBayes
from pyei.r_by_c import RowByColumnEI

### CLEANING AND MERGING REGISTRATION AND 2024 VOTERFILE ###
print("CSV CLEANING AND MERGING")

# Define chunk size for reading in pieces
CHUNK_SIZE = 100000  # Adjust this based on available memory

# Columns to keep and their target data types
columns_to_keep = {
    "PRECINCT": "int", 
    "CONGRESS": "int", 
    "ASSEMBLY": "int", 
    "SENATE": "int", 
    "COMMISSION": "int",
    "PARTY_REG": "str", 
    "REGISTRATION_NUM": "int"
}

# Initialize an empty DataFrame to store cleaned data
cleaned_chunks = []

# Process the registration file in chunks
for chunk in pd.read_csv('data/registration.csv', chunksize=CHUNK_SIZE, low_memory=False):
    # Drop unnecessary columns
    chunk = chunk[columns_to_keep.keys()]

    # Convert columns to the desired data types
    for col, dtype in columns_to_keep.items():
        if dtype == "int":
            # Handle leading zeros and mixed types
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("Int64")
        elif dtype == "str":
            chunk[col] = chunk[col].astype("str")

    # Append the cleaned chunk to the list
    cleaned_chunks.append(chunk)

# Concatenate all cleaned chunks into a single DataFrame
cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)

# Save the cleaned registration data
cleaned_df.to_csv("data/registration_fixed.csv", index=False)
print("Cleaned registration data saved to 'data/registration_fixed.csv'.")

# Ensure idnumber in voters.csv is also an integer
voters_df = pd.read_csv('data/voters.csv', dtype={'idnumber': 'str'})

# Convert idnumber to integer
voters_df['idnumber'] = pd.to_numeric(voters_df['idnumber'], errors='coerce').astype("Int64")

# Save the cleaned voters data
voters_df.to_csv("data/voters_fixed.csv", index=False)
print("Cleaned voters data saved to 'data/voters_fixed.csv'.")

# Continue with the merge with voters_fixed.csv
merged_chunks = []

# Read the cleaned registration file in chunks
for i in range(0, len(cleaned_df), CHUNK_SIZE):
    # Create a chunk from the DataFrame
    chunk = cleaned_df.iloc[i:i + CHUNK_SIZE]

    # Merge with the voters DataFrame
    merged_chunk = pd.merge(chunk, voters_df,
                            left_on='REGISTRATION_NUM', right_on='idnumber', how='inner')

    # Append the merged chunk to the list
    merged_chunks.append(merged_chunk)

# Concatenate all merged chunks into a single DataFrame
voters_df = pd.concat(merged_chunks, ignore_index=True)

# Save the result to a new CSV
voters_df.to_csv('data/filtered_voters.csv', index=False)
print("Filtered join completed and saved to 'filtered_voters.csv'.")

# Final preview
print(voters_df.head())
print(voters_df.tail())

### PRECINCT SHAPEFILE ###
print("\nPRECINCT SHAPEFILE CLEANING")

# Suppress only FutureWarnings - I do not care about ChainedAssignmentErrors in maup
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the shapefile
original_gdf = gpd.read_file("data/shapefiles/precinct_p.shp")
print(original_gdf)
print(original_gdf.loc[[985, 987]])

# Information
print("original_gdf.info:\n")
original_gdf.info(verbose=True)

print("\nmaup doctor on unrepaired\n")
maup.doctor(original_gdf)
print("\nEnd maup doctor on unrepaired\n")

# Plot original gdf
original_gdf.plot()
plt.title(f"original_gdf.png")
plt.savefig("figs/original_gdf.png")
print("Saved figs/original_gdf.png\n")

# Smart repair
print("Running smart repair on original gdf\n")
repaired_gdf = smart_repair(original_gdf)
print("\nEnd smart repair on original gdf\n")
print(repaired_gdf)

# Check that repair succeeded:
print("maup doctor on repaired gdf")
maup.doctor(repaired_gdf)
print("End maup doctor on repaired gdf\n")

# Dual graph sanity check
print("Loading repaired gdf graph")
# graph = Graph.from_file(PATH)
graph = Graph.from_geodataframe(repaired_gdf)
print("Repaired gdf graph loaded\n")

# Information about graph
print(f"Is the dual graph connected? {nx.is_connected(graph)}")
print(f"Is the dual graph planar? {nx.is_planar(graph)}")
print(f"Number of Nodes: {len(graph.nodes())}")
print(f"Number of Edges: {len(graph.edges())}\n")

# Plot repair one
repaired_gdf.plot()
plt.title(f"repaired_gdf.png")
plt.savefig("figs/repaired.png")
print("Saved figs/repaired.png\n")

# Drop the Shape_Area column, we don't need area in this analysis
print("Getting rid of area\n")
no_area_gdf = repaired_gdf.drop(columns=['Shape_Area'])
print(no_area_gdf)

# Check the geometry column
print("Geometry type:", no_area_gdf['geometry'].geom_type.unique())

# Check if 'PREC' is unique
print("Getting rid of duplicate PREC")
is_unique = no_area_gdf['PREC'].is_unique
print("Is 'PREC' unique?:", is_unique)

# Identify rows with duplicate PREC values
duplicates = no_area_gdf[no_area_gdf.duplicated(subset='PREC', keep=False)]
print("Duplicate PREC rows:\n", duplicates)

# Filter the two rows with PREC value 7908
precinct_7908 = no_area_gdf[no_area_gdf['PREC'] == 7908]

# Merge the geometries of the precinct (using unary_union)
print("Union weird precinct")
merged_geometry = precinct_7908.geometry.unary_union
# simplified_geometry = unary_union([geom for geom in merged_geometry.geoms if geom.area > 0.0001])
print("Done unioning weird precinct")

# Update the GeoDataFrame with the merged geometry (keep other attributes intact)
precincts_gdf = no_area_gdf.copy()
print("Merge unioned weird precinct with gdf")
precincts_gdf.loc[precincts_gdf['PREC'] == 7908, 'geometry'] = merged_geometry
print("Weird precinct merged back in")

# Deal with duplicate
duplicates = precincts_gdf[precincts_gdf['PREC'] == 7908]
print("Now we have duplicates:\n", duplicates)
print("Drop duplicate")
precincts_gdf = precincts_gdf.drop(index=duplicates.index[1])  # Drop the second instance
print("Recalculate Shape_Leng")
precincts_gdf['Shape_Leng'] = precincts_gdf['geometry'].length

# Plot the updated GeoDataFrame
precincts_gdf.plot()
plt.title(f"precincts_gdf.png")
plt.savefig("figs/merged.png")
print("\nSaved figs/merged.png")

# Use maup to ensure all geometries are valid
print("\nmaup doctor on merged gdf")
maup.doctor(precincts_gdf)
print("End maup doctor on merged gdf")

# Check dual graph
print("\nGraph sanity check merged gdf")
graph = Graph.from_geodataframe(precincts_gdf)
print(f"Graph columns: {graph.nodes()[0].keys()}\n")
print(f"Is the dual graph connected? {nx.is_connected(graph)}")

# Save the repaired shapefile
precincts_gdf.to_file("data/precinct_p_fixed.shp")

### ADDING VOTERFILE INFO TO PRECINCT SHAPEFILE ###
print("VOTERFILE & PRECINCT SHAPEFILE")

# Read in voterfile
# voters_df = pd.read_csv("data/filtered_voters.csv")
print("voters_df:\n", voters_df.columns)
# precincts_gdf = gpd.read_file("data/precinct_p_fixed.shp")
print("precincts_gdf:\n", precincts_gdf.columns)

# Check for unique PREC in both datasets
print("Number of unique precincts in precincts_gdf:", len(precincts_gdf['PREC'].unique()))
print("Number of unique precincts in voters_df:", len(voters_df['PRECINCT'].unique()))

# Identify missing precincts in either dataset
precincts_missing_voters = set(precincts_gdf['PREC']) - set(voters_df['PRECINCT'])
voters_missing_precincts = set(voters_df['PRECINCT']) - set(precincts_gdf['PREC'])

print("Number of precincts in shapefile but not in CSV:", len(precincts_missing_voters))
# This is expected to be high, because there are plenty of precincts with very low/no pop
# There are 195 precincts without any people who voted in 2024

print("Precincts in CSV but not in shapefile:", voters_missing_precincts)
# This should be very low, because every voter should have a precinct
# Voters without a precinct are assigned precinct "9996", and also don't have assigned districts

# Thus, we will remove those 37 voters out of our analysis as they aren't spatially located
voters_df_filtered = voters_df[voters_df['PRECINCT'].isin(precincts_gdf['PREC'])]
print(f"Filtered voter records: {len(voters_df_filtered)}")
print("voters_df_filtered:\n", voters_df_filtered.columns)

# Aggregate voter data
party_counts = (
    voters_df_filtered
    .groupby(['PRECINCT', 'PARTY_REG'])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
party_counts.rename(columns={'PRECINCT': 'precinct'}, inplace=True)

# Merge aggregated data with the shapefile
print("party_counts:\n", party_counts.columns)
print("weird column:", party_counts['   '].unique())
print("weird voters:\n", voters_df_filtered[voters_df_filtered['PARTY_REG'] == '   '])
# precinct_7908 = no_area_gdf[no_area_gdf['PREC'] == 7908]

# Select columns to keep and rename them
precincts_gdf = precincts_gdf[['PREC', 'ASSEMBLY', 'SENATE', 'CONGRESS', 'COMMISSION', 'geometry']]
print("precincts_gdf:\n", precincts_gdf.columns)

# Rename columns
precincts_gdf = precincts_gdf.rename(columns={
    "PREC": "precinct",
    "ASSEMBLY": "assembly",
    "SENATE": "senate",
    "CONGRESS": "congress",
    "COMMISSION": "commission"
})
print("precincts_gdf:\n", precincts_gdf.columns)

aggregated_gdf = precincts_gdf.merge(
    party_counts,
    on="precinct",
    how="left"
)
print("aggregated_gdf:\n", aggregated_gdf.columns)

# Select only required columns
aggregated_columns = ['precinct', 'assembly', 'senate', 'congress', 'commission', 'geometry'] + list(party_counts.columns.drop('precinct'))
aggregated_gdf = aggregated_gdf[aggregated_columns]

# Deal with NAs
# Add counts from '   ' and 'nan' to the 'NP ' column
aggregated_gdf['NP '] += aggregated_gdf[['   ', 'nan']].sum(axis=1)

# Drop the '   ' and 'nan' columns
aggregated_gdf = aggregated_gdf.drop(columns=['   ', 'nan'])

# Fill NAs in the party counts
aggregated_gdf = aggregated_gdf.fillna(0)
print("aggregated_gdf:\n", aggregated_gdf.columns)

# Sanity check
assert len(aggregated_gdf) == len(aggregated_gdf['precinct'].unique()), "Duplicated rows detected!"

# Save the aggregated shapefile
aggregated_gdf_no_geom = aggregated_gdf.drop(columns=['geometry'])
aggregated_gdf_no_geom.to_csv("data/aggregated_precincts.csv", index=False)
aggregated_gdf.to_file("data/aggregated_precincts.shp")

### CAST VOTE RECORD ###
aggregated_gdf = gpd.read_file('data/aggregated_precincts.shp')

# Track total rows processed and dropped
total_rows_processed = 0
total_rows_dropped = 0

# Initialize an empty DataFrame to accumulate precinct summaries
precinct_summary = pd.DataFrame()

# Process precinct column using regex
def clean_precinct(precinct):
    # Extract precinct number from format like "1234 (1234|00)"
    match = re.match(r'^(\d+)', str(precinct))
    return int(match.group(1)) if match else None

# Read CSV, skipping first 4 rows of headers
for chunk in pd.read_csv('data/cast_vote.csv', header=4, chunksize=CHUNK_SIZE, low_memory=False):
    # Select specific columns (index 6 and 16-20)
    selected_columns = [chunk.columns[i] for i in [6, 16, 17, 18, 19, 20]]
    chunk = chunk[selected_columns]
    
    # Rename columns
    column_names = ["precinct", "harris", "oliver", "skousen", "trump", "none"]
    chunk.columns = column_names
    
    # Convert columns to int, filling NAs with 0
    numeric_columns = column_names[1:]
    for col in numeric_columns:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0).astype(int)
    
    # Create 'other' column by summing specified columns
    chunk['other'] = chunk['oliver'] + chunk['skousen'] + chunk['none']
    
    # Drop these columns after summing
    chunk = chunk.drop(columns=['oliver', 'skousen', 'none'])
    
    # Track rows that will be dropped
    rows_before = len(chunk)
    chunk['precinct'] = chunk['precinct'].apply(clean_precinct)
    
    # Drop NA rows and track dropped rows
    rows_dropped = chunk[chunk['precinct'].isna()]
    if not rows_dropped.empty:
        print(f"Rows dropped in this chunk (precinct could not be parsed):")
        print(rows_dropped)
    
    chunk = chunk.dropna(subset=['precinct'])
    chunk['precinct'] = chunk['precinct'].astype(int)
    
    # Track rows
    rows_in_chunk = len(chunk)
    total_rows_processed += rows_before
    total_rows_dropped += (rows_before - rows_in_chunk)
    
    # Group by precinct and sum votes for this chunk
    chunk_summary = chunk.groupby('precinct').agg({
        'harris': 'sum',
        'trump': 'sum',
        'other': 'sum'
    }).reset_index()
    
    # Calculate total votes per precinct
    chunk_summary['total_vote'] = (
        chunk_summary['harris'] + 
        chunk_summary['trump'] + 
        chunk_summary['other']
    )
    
    # Accumulate summaries
    precinct_summary = pd.concat([precinct_summary, chunk_summary], ignore_index=True)

# Aggregate the final summary
precinct_summary = precinct_summary.groupby('precinct').agg({
    'harris': 'sum',
    'trump': 'sum',
    'other': 'sum',
    'total_vote': 'sum'
}).reset_index()

# Again, drop voters in the 9996 "precinct" because they are not spatially organized
precinct_summary = precinct_summary[precinct_summary['precinct'] != 9996]

# Check for any vote count precincts not in aggregated_gdf
unmatched_precincts = precinct_summary[~precinct_summary['precinct'].isin(aggregated_gdf['precinct'])]
if not unmatched_precincts.empty:
    raise ValueError(f"Precincts in vote counts not found in aggregated_gdf: {unmatched_precincts['precinct'].tolist()}")

# Save the precinct
precinct_summary.to_csv("data/cast_vote_fixed.csv", index=False)

# Merge with left join (keeping all rows from aggregated_gdf)
# Fill missing vote count rows with 0
final_gdf = aggregated_gdf.merge(precinct_summary, on='precinct', how='left')
final_gdf[['harris', 'trump', 'other', 'total_vote']] = final_gdf[['harris', 'trump', 'other', 'total_vote']].fillna(0)

print("final_gdf:\n", final_gdf.columns)
print("final_gdf head:\n", final_gdf.head())
print(f"Total rows processed: {total_rows_processed}")
print(f"Total rows dropped: {total_rows_dropped}")

# Save the final shapefile
final_gdf_no_geom = final_gdf.drop(columns=['geometry'])
final_gdf_no_geom.to_csv("data/final_precincts.csv", index=False)
final_gdf.to_file("data/final_precincts.shp")
