import pandas as pd
import io

filename = 'smhi_pred.csv'

# Automatically find the header row (starts with 'year')
with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()
header_idx = next(i for i, line in enumerate(lines) if line.strip().startswith('year'))

# Read the actual data part, using semicolon as the separator
df = pd.read_csv(io.StringIO(''.join(lines[header_idx:])), sep=';')

print(df.head())
print(df.columns.tolist())

# Save the cleaned DataFrame to a new CSV file
df.to_csv('smhi_pred_cleaned.csv', index=False)
