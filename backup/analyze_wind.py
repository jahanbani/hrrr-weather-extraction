#!/usr/bin/env python3
import pandas as pd

# Read the wind output file
df = pd.read_parquet('wind_output_special_2022_12_to_2023_2.parquet')

print('Wind Power Output Analysis:')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Time range: {df.index.min()} to {df.index.max()}')

print('\nMax power per farm:')
for col in df.columns:
    max_power = df[col].max()
    print(f'  {col}: {max_power:.3f} MW')

print(f'\nOverall max: {df.max().max():.3f} MW')
print(f'Overall mean: {df.mean().mean():.3f} MW')

# Check if any farm reaches 1 MW
max_per_farm = df.max()
print(f'\nFarms reaching 1 MW: {sum(max_per_farm >= 1.0)} out of {len(max_per_farm)}')
print(f'Farms with max >= 0.9 MW: {sum(max_per_farm >= 0.9)} out of {len(max_per_farm)}') 