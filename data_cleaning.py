import warnings
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from gerrychain import Graph
import maup
from maup import smart_repair, resolve_overlaps
from shapely.ops import unary_union

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
    "PARTY_REG": "str", 
    "RES_STREET_NUM": "str", 
    "RES_DIRECTION": "str", 
    "RES_STREET_NAME": "str", 
    "RES_ADDRESS_TYPE": "str", 
    "RES_UNIT": "str", 
    "RES_CITY": "str", 
    "RES_STATE": "str", 
    "RES_ZIP_CODE": "int", 
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
precincts_gdf = precincts_gdf[['PREC', 'ASSEMBLY', 'SENATE', 'CONGRESS', 'geometry']]
print("precincts_gdf:\n", precincts_gdf.columns)

# Rename columns
precincts_gdf = precincts_gdf.rename(columns={
    "PREC": "precinct",
    "ASSEMBLY": "assembly",
    "SENATE": "senate",
    "CONGRESS": "congress",
})
print("precincts_gdf:\n", precincts_gdf.columns)

final_gdf = precincts_gdf.merge(
    party_counts,
    on="precinct",
    how="left"
)
print("final_gdf:\n", final_gdf.columns)

# Select only required columns
final_columns = ['precinct', 'assembly', 'senate', 'congress', 'geometry'] + list(party_counts.columns.drop('precinct'))
final_gdf = final_gdf[final_columns]

# Deal with NAs
final_gdf = final_gdf.rename(columns={
    '   ': "NaP" # this, to me, stands for Not a Party
})
final_gdf = final_gdf.fillna(0)
print("final_gdf:\n", final_gdf.columns)

# Save the final shapefile
final_gdf_no_geom = final_gdf.drop(columns=['geometry'])
final_gdf_no_geom.to_csv("data/aggregated_precincts.csv", index=False)
final_gdf.to_file("data/aggregated_precincts.shp")

# Optional: print summary of party registration and voting methods
print("Party registration counts per precinct:")
print(party_counts.sum(axis=0))
