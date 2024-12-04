import pandas as pd

# # Define chunk size for reading in pieces
chunk_size = 100000  # You can adjust this based on available memory

# Initialize an empty list to store merged chunks
merged_chunks = []

# Read the registration file in chunks
for chunk in pd.read_csv('registration.csv', dtype={'REGISTRATION_NUM': 'str'}, chunksize=chunk_size):
    # Read the voters file and merge with the current chunk
    voters_df = pd.read_csv('voters.csv', dtype={'idnumber': 'str'})
    merged_chunk = pd.merge(chunk, voters_df, left_on='REGISTRATION_NUM', right_on='idnumber',
                            how='inner')
    
    # Append the merged chunk to the list
    merged_chunks.append(merged_chunk)

# Concatenate all merged chunks into a single DataFrame
final_df = pd.concat(merged_chunks, ignore_index=True)

# Save the result to a new CSV
final_df.to_csv('filtered_voters.csv', index=False)

# print("Filtered left join completed and saved to 'filtered_voters.csv'.")

# final_df = pd.read_csv('filtered_voters.csv')
print(f"DEBUG: final_df:\n{final_df.head()}")
print(f"DEBUG: final_df:\n{final_df.tail()}")
