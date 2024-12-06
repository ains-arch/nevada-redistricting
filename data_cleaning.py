import warnings
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from gerrychain import Graph
import maup
from maup import smart_repair, resolve_overlaps
from shapely.ops import unary_union

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
final_df = pd.concat(merged_chunks, ignore_index=True)

# Save the result to a new CSV
final_df.to_csv('data/filtered_voters.csv', index=False)
print("Filtered join completed and saved to 'filtered_voters.csv'.")

# Final preview
print(final_df.head())
print(final_df.tail())

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
merged_gdf = no_area_gdf.copy()
print("Merge unioned weird precinct with gdf")
merged_gdf.loc[merged_gdf['PREC'] == 7908, 'geometry'] = merged_geometry
print("Weird precinct merged back in")

# Deal with duplicate
duplicates = merged_gdf[merged_gdf['PREC'] == 7908]
print("Now we have duplicates:\n", duplicates)
print("Drop duplicate")
merged_gdf = merged_gdf.drop(index=duplicates.index[1])  # Drop the second instance
print("Recalculate Shape_Leng")
merged_gdf['Shape_Leng'] = merged_gdf['geometry'].length

# Plot the updated GeoDataFrame
merged_gdf.plot()
plt.savefig("figs/merged.png")
print("\nSaved figs/merged.png")

# Use maup to ensure all geometries are valid
print("\nmaup doctor on merged gdf")
maup.doctor(merged_gdf)
print("End maup doctor on merged gdf")

# Check dual graph
print("\nGraph sanity check merged gdf")
graph = Graph.from_geodataframe(merged_gdf)

# Save the repaired shapefile
merged_gdf.to_file("data/precinct_p_fixed.shp")
