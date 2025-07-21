
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, gamma, genextreme

# Read and clean data
df = pd.read_csv('C:/Users/123/Downloads/policy_lab/policy_lab_2025/event_and_magnitudes_analysis/historical_precipitation_fixed.csv')
df['Date'] = pd.to_datetime(df['Representative date'], dayfirst=True)
df['Precip'] = pd.to_numeric(df['Rainfall'], errors='coerce')
df = df[['Date', 'Precip']].sort_values('Date').reset_index(drop=True)
df['year'] = df['Date'].dt.year

# Add season and decade columns
# Assign each row to a meteorological season and decade period
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'
df['season'] = df['Date'].dt.month.apply(get_season)
df['decade'] = pd.cut(df['year'], bins=[2004, 2014, 2025], labels=['2005-2014', '2015-2025'], right=True)

# Event detection function
# Detect consecutive days above a given precipitation threshold as an event
def find_events(subdf, threshold):
    events = []
    in_event = False
    start = None
    idxs = subdf.index
    for i in idxs:
        val = subdf.at[i, 'Precip']
        if not in_event and val >= threshold:
            in_event = True
            start = i
        if in_event and (val < threshold or i == idxs[-1]):
            end = i if val >= threshold else i-1
            event_rows = subdf.loc[start:end]
            events.append({
                'start_date': event_rows.iloc[0]['Date'],
                'end_date': event_rows.iloc[-1]['Date'],
                'duration': (event_rows.iloc[-1]['Date'] - event_rows.iloc[0]['Date']).days + 1,
                'total_precip': event_rows['Precip'].sum()
            })
            in_event = False
    return pd.DataFrame(events)

# Grouped analysis (by decade and season)
# For each decade and each season, extract events, intervals, and event magnitudes
results = {}
for dec in df['decade'].dropna().unique():
    for sea in ['Winter', 'Spring', 'Summer', 'Autumn']:
        subdf = df[(df['decade'] == dec) & (df['season'] == sea)]
        for label, thresh in [('heavy', 10), ('extreme', 20)]:
            key = f'{label}_{dec}_{sea}'
            events = find_events(subdf, threshold=thresh)
            intervals = pd.Series(dtype='timedelta64[ns]')
            if len(events) > 1:
                intervals = events['start_date'].iloc[1:].reset_index(drop=True) - events['end_date'].iloc[:-1].reset_index(drop=True)
            magnitudes = events['total_precip'] if len(events) > 0 else pd.Series(dtype=float)
            results[key] = {
                'events': events,
                'intervals': intervals,
                'magnitudes': magnitudes,
                'season_length': len(subdf),
                'first_day': subdf['Date'].min() if not subdf.empty else None,
                'last_day': subdf['Date'].max() if not subdf.empty else None
            }

# Fit distributions for intervals and magnitudes
# For each group, fit exponential/gamma to intervals, and GEV to event magnitudes
def fit_and_report(data, dist='expon'):
    data = np.asarray(data)
    if len(data) == 0 or np.any(np.isnan(data)):
        return None
    if dist == 'expon':
        params = expon.fit(data)
    elif dist == 'gamma':
        params = gamma.fit(data)
    elif dist == 'gev':
        if len(data) < 2: return None
        params = genextreme.fit(data)
    else:
        raise ValueError('Unsupported distribution')
    return params

fit_results = {}
for key, d in results.items():
    intervals = d['intervals'].dt.days.values if not d['intervals'].empty else np.array([])
    mags = d['magnitudes'].values if not d['magnitudes'].empty else np.array([])
    fit_results[key] = {
        'exp_interval': fit_and_report(intervals, 'expon') if len(intervals) > 0 else None,
        'gamma_interval': fit_and_report(intervals, 'gamma') if len(intervals) > 0 else None,
        'gev_magnitude': fit_and_report(mags, 'gev') if len(mags) > 1 else None
    }

# Print one example group fit and summary
example_key = 'heavy_2005-2014_Summer'
print(f"\n=== {example_key} ===")
print("Events Table:\n", results[example_key]['events'])
print("Intervals (days):", results[example_key]['intervals'].dt.days.values if not results[example_key]['intervals'].empty else [])
print("Magnitudes:", results[example_key]['magnitudes'].values if not results[example_key]['magnitudes'].empty else [])
print("Exp fit (interval):", fit_results[example_key]['exp_interval'])
print("Gamma fit (interval):", fit_results[example_key]['gamma_interval'])
print("GEV fit (magnitude):", fit_results[example_key]['gev_magnitude'])

# Plot histograms and fitted distributions
# Visualize interval and magnitude distributions, overlay fitted distributions
intervals = results[example_key]['intervals'].dt.days.values if not results[example_key]['intervals'].empty else []
if len(intervals) > 0:
    plt.hist(intervals, bins=10, alpha=0.6, label='Intervals')
    x = np.linspace(0, max(intervals), 100)
    if fit_results[example_key]['exp_interval'] is not None:
        plt.plot(x, len(intervals)*(np.diff(np.histogram(intervals, bins=10)[1])[0])*expon.pdf(x, *fit_results[example_key]['exp_interval']), label='Exp fit')
    if fit_results[example_key]['gamma_interval'] is not None:
        plt.plot(x, len(intervals)*(np.diff(np.histogram(intervals, bins=10)[1])[0])*gamma.pdf(x, *fit_results[example_key]['gamma_interval']), label='Gamma fit')
    plt.legend()
    plt.title(f"{example_key} - Event intervals")
    plt.xlabel("Interval (days)")
    plt.ylabel("Count")
    plt.show()

magnitudes = results[example_key]['magnitudes'].values if not results[example_key]['magnitudes'].empty else []
if len(magnitudes) > 1 and fit_results[example_key]['gev_magnitude'] is not None:
    plt.hist(magnitudes, bins=10, alpha=0.6, label='Event magnitude')
    x = np.linspace(min(magnitudes), max(magnitudes), 100)
    c, loc, scale = fit_results[example_key]['gev_magnitude']
    plt.plot(x, len(magnitudes)*(np.diff(np.histogram(magnitudes, bins=10)[1])[0])*genextreme.pdf(x, c, loc, scale), label='GEV fit')
    plt.legend()
    plt.title(f"{example_key} - Event magnitudes")
    plt.xlabel("Total precipitation (mm)")
    plt.ylabel("Count")
    plt.show()

# Export summary table (seasonal info, event count, etc.)
# Export summary table with seasonal statistics for each group
summary = []
for dec in df['decade'].dropna().unique():
    for sea in ['Winter', 'Spring', 'Summer', 'Autumn']:
        for label in ['heavy', 'extreme']:
            key = f'{label}_{dec}_{sea}'
            d = results.get(key, None)
            if d is not None:
                n_events = len(d['events'])
                season_length = d['season_length']
                first_day = d['first_day']
                last_day = d['last_day']
            else:
                n_events = 0
                subdf = df[(df['decade'] == dec) & (df['season'] == sea)]
                season_length = len(subdf)
                first_day = subdf['Date'].min() if not subdf.empty else None
                last_day = subdf['Date'].max() if not subdf.empty else None
            summary.append({
                'group': key,
                'n_events': n_events,
                'has_event': n_events > 0,
                'season_length': season_length,
                'first_day': first_day,
                'last_day': last_day
            })

summary_df = pd.DataFrame(summary)
summary_df.to_csv('seasonal_event_summary.csv', index=False)

# Export all event tables (optional)
# Export detailed event tables for each group
for key, d in results.items():
    d['events'].to_csv(f'{key}_events.csv', index=False)

# Export intervals and magnitudes for each group
# Export interval tables with contextual info for each group
for key, d in results.items():
    events = d['events']
    intervals = d['intervals']
    if not intervals.empty and len(events) > 1:
        intervals_df = pd.DataFrame({
            'prev_end_date': events['end_date'].iloc[:-1].values,
            'prev_magnitude': events['total_precip'].iloc[:-1].values,
            'next_start_date': events['start_date'].iloc[1:].values,
            'next_magnitude': events['total_precip'].iloc[1:].values,
            'interval_days': intervals.dt.days.values
        })
        intervals_df.to_csv(f'{key}_intervals.csv', index=False)

# Count yearly/seasonal events for each year and season (for completeness)
event_counts = []
for label, thresh in [('heavy', 10), ('extreme', 20)]:
    for year in sorted(df['year'].unique()):
        for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
            # For each year and season, count the number of detected events
            mask = (df['year'] == year) & (df['season'] == season)
            subdf = df[mask]
            events = find_events(subdf, threshold=thresh)
            event_counts.append({
                'year': year,
                'season': season,
                'type': label,
                'n_events': len(events)
            })
event_counts_df = pd.DataFrame(event_counts)
event_counts_df.to_csv('yearly_seasonal_event_counts.csv', index=False)

print("\nAll results have been processed and exported.")
