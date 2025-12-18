import pandas as pd
import os

# Read all CSV files
print("Reading CSV files...")
ca_data = pd.read_csv('data/CAvideos.csv')
gb_data = pd.read_csv('data/GBvideos.csv')
in_data = pd.read_csv('data/INvideos.csv')
us_data = pd.read_csv('data/USvideos.csv')

# Add country column to each dataset
ca_data['country'] = 'CA'
gb_data['country'] = 'GB'
in_data['country'] = 'IN'
us_data['country'] = 'US'

# Merge all datasets
print("Merging datasets...")
merged_data = pd.concat([ca_data, gb_data, in_data, us_data], ignore_index=True)

# Save merged data
output_path = 'data/merged_youtube_data.csv'
merged_data.to_csv(output_path, index=False)

print(f"Merged data saved to {output_path}")
print(f"Total rows: {len(merged_data)}")
print(f"Total columns: {len(merged_data.columns)}")
print(f"\nDataset Breakdown:")
print(f"CA videos: {len(ca_data)}")
print(f"GB videos: {len(gb_data)}")
print(f"IN videos: {len(in_data)}")
print(f"US videos: {len(us_data)}")
