import pandas as pd
import geopandas as gpd

# Define chunk size for reading in pieces
chunk_size = 100000  # Adjust this based on available memory

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
for chunk in pd.read_csv('data/registration.csv', chunksize=chunk_size, low_memory=False):
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
chunk_size = 100000  # Reset chunk size for merge
merged_chunks = []

# Read the cleaned registration file in chunks
for i in range(0, len(cleaned_df), chunk_size):
    # Create a chunk from the DataFrame
    chunk = cleaned_df.iloc[i:i + chunk_size]

    # Merge with the voters DataFrame
    merged_chunk = pd.merge(chunk, voters_df, left_on='REGISTRATION_NUM', right_on='idnumber', how='inner')
    
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

# Load the shapefile
gdf = gpd.read_file("data/shapefiles/precinct_p.shp")
# print("original max", gdf['Shape_Area'].max())

# Check invalid geometries
print("valid?", gdf.is_valid.all())
invalid_rows = [572, 677, 776, 967, 994]
print("Invalid geometries\n", gdf.iloc[invalid_rows])

# Fix invalid geometries
gdf['geometry'] = gdf['geometry'].buffer(0)

# Verify if there are still invalid geometries
print("Valid count:", gdf.is_valid.value_counts())

# Check for remaining invalid geometries
if not gdf.is_valid.all():
    print("Still contains invalid geometries")
else:
    print("All geometries are valid")

# Print datatypes
print("gdf.dtypes:", gdf.dtypes)
# print("shape_area max:", gdf['Shape_Area'].max())

# Drop the Shape_Area column, we don't need area in this analysis
gdf = gdf.drop(columns=['Shape_Area'])

# Save the repaired shapefile without the Shape_Area column
gdf.to_file("data/precinct_p_fixed.shp")
