import pandas as pd
import io
import os

filename = r"F:\Policy_2025_Lab\heavy_rain_scenario\r10mm_rcp45_ANN_yr_1951_2100_gavleborgs_lan.csv"

# Automatically find the header row (starts with 'year')
with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()
header_idx = next(i for i, line in enumerate(lines) if line.strip().startswith('year'))

# Read the actual data part, using semicolon as the separator
df = pd.read_csv(io.StringIO(''.join(lines[header_idx:])), sep=';')

print(df.head())
print(df.columns.tolist())

# Save the cleaned DataFrame to the same folder as the original file
folder = os.path.dirname(filename)
output_path = os.path.join(folder, 'smhi_RCP4.5_10MM.csv')
df.to_csv(output_path, index=False)

print(f"File exported to: {output_path}")
