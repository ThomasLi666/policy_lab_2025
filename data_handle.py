import pandas as pd
import io

filename = 'smhi-opendata_5_116170_20250703_123715.csv'  # Change to your actual file name

# 1. Automatically locate the header row (for raw SMHI CSV)
with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()
header_idx = next(i for i, line in enumerate(lines) if 'Från Datum' in line)
df = pd.read_csv(io.StringIO(''.join(lines[header_idx:])), sep=';')

# 2. Keep date and precipitation columns, auto-detect column names
date_col = [c for c in df.columns if 'dygn' in c.lower() or 'date' in c.lower() or 'datum' in c.lower()][0]
pr_col = [c for c in df.columns if 'nederbörd' in c.lower() or 'precip' in c.lower() or 'mm' in c.lower()][0]
df = df[[date_col, pr_col]].rename(columns={date_col: 'date', pr_col: 'precip'})

# 3. Standardize data types
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['precip'] = pd.to_numeric(df['precip'], errors='coerce')

# 4. Add season label
def season_from_month(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
df['season'] = df['month'].apply(season_from_month)

# 5. Mark heavy/extreme precipitation days
df['heavy'] = df['precip'] > 10
df['extreme'] = df['precip'] > 20

# 6. Annual statistics for extreme precipitation days
annual_stats = df.groupby('year').agg(
    n_heavy=('heavy', 'sum'),
    n_extreme=('extreme', 'sum'),
    max_precip=('precip', 'max'),
    mean_precip=('precip', 'mean'),
    total_days=('precip', 'count')
).reset_index()
print('\n--- Annual extreme precipitation statistics ---')
print(annual_stats.head())

# 7. Seasonal statistics for extreme precipitation days
seasonal_stats = df.groupby(['year', 'season']).agg(
    n_heavy=('heavy', 'sum'),
    n_extreme=('extreme', 'sum'),
    mean_precip=('precip', 'mean'),
    total_days=('precip', 'count')
).reset_index()
print('\n--- Annual + seasonal extreme precipitation statistics ---')
print(seasonal_stats.head())

# 8. Export detailed lists of heavy/extreme precipitation days
heavy_events = df[df['heavy']]
extreme_events = df[df['extreme']]
heavy_events.to_csv('all_heavy_days.csv', index=False)
extreme_events.to_csv('all_extreme_days.csv', index=False)

# 9. Arrays for Bayesian modeling
n_heavy_seq = annual_stats['n_heavy'].values  # Annual heavy precipitation event counts
n_extreme_seq = annual_stats['n_extreme'].values

# 10. Export results
annual_stats.to_csv('annual_extreme_stats.csv', index=False)
seasonal_stats.to_csv('seasonal_extreme_stats.csv', index=False)

print('\nAll extreme precipitation data processing and export completed!')
