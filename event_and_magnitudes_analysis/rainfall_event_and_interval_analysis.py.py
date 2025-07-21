import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, gamma, genextreme

# ========== 1. Read and clean data ==========
df = pd.read_csv('C:/Users/123/Downloads/policy_lab/policy_lab_2025/event_and_magnitudes_analysis/historical_precipitation_fixed.csv')
df['Date'] = pd.to_datetime(df['Representative date'], dayfirst=True)
df['Precip'] = pd.to_numeric(df['Rainfall'], errors='coerce')
df = df[['Date', 'Precip']].sort_values('Date').reset_index(drop=True)
df['year'] = df['Date'].dt.year

# ========== 2. Add season and decade columns ==========
# For cross-year winter season: assign "Winter YYYY/YYYY+1" label
def get_season_and_winter_label(date):
    y, m = date.year, date.month
    if m == 12:
        return 'Winter', f"{y}/{y+1}"
    elif m == 1 or m == 2:
        return 'Winter', f"{y-1}/{y}"
    elif m in [3,4,5]:
        return 'Spring', f"{y}"
    elif m in [6,7,8]:
        return 'Summer', f"{y}"
    else:
        return 'Autumn', f"{y}"

df['season'], df['season_group'] = zip(*df['Date'].apply(get_season_and_winter_label))
# decade assignment
def assign_decade(row):
    # Use the start year for winter, normal year for others
    yr = int(row['season_group'][:4]) if row['season'] == 'Winter' else int(row['season_group'])
    if 2005 <= yr <= 2014:
        return '2005-2014'
    elif 2015 <= yr <= 2025:
        return '2015-2025'
    else:
        return np.nan
df['decade'] = df.apply(assign_decade, axis=1)

# ========== 3. Event detection function ==========
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

# ========== 4. Grouped analysis (cross-year winter) ==========
results = {}
for dec in ['2005-2014', '2015-2025']:
    # find all unique group labels in this decade for all seasons
    available_groups = df.loc[df['decade']==dec, ['season', 'season_group']].drop_duplicates()
    for _, (season, group_label) in available_groups.iterrows():
        subdf = df[(df['decade']==dec) & (df['season']==season) & (df['season_group']==group_label)]
        for label, thresh in [('heavy', 10), ('extreme', 20)]:
            key = f'{label}_{dec}_{season}_{group_label}'
            events = find_events(subdf, threshold=thresh)
            intervals = pd.Series(dtype='timedelta64[ns]')
            if len(events) > 1:
                tmp_intervals = (
                    events['start_date'].iloc[1:].reset_index(drop=True) -
                    events['end_date'].iloc[:-1].reset_index(drop=True)
                )
                max_interval = subdf['Date'].max() - subdf['Date'].min() + pd.Timedelta(days=1)
                intervals = tmp_intervals[tmp_intervals <= max_interval]
            magnitudes = events['total_precip'] if len(events) > 0 else pd.Series(dtype=float)
            results[key] = {
                'events': events,
                'intervals': intervals,
                'magnitudes': magnitudes,
                'season_length': len(subdf),
                'first_day': subdf['Date'].min() if not subdf.empty else None,
                'last_day': subdf['Date'].max() if not subdf.empty else None
            }

# ========== 5. Fit distributions for intervals and magnitudes ==========
def fit_and_report(data, dist='expon'):
    data = np.asarray(data)
    if len(data) == 0 or np.any(np.isnan(data)):
        return None
    if dist == 'expon':
        params = expon.fit(data)
    elif dist == 'gamma':
        if len(data) < 2:
            return None
        try:
            params = gamma.fit(data)
        except Exception as e:
            print(f"Fit error for gamma on data {data}: {e}")
            return None
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

# ========== 6. Export results ==========
summary = []
for key, d in results.items():
    summary.append({
        'group': key,
        'n_events': len(d['events']),
        'has_event': len(d['events']) > 0,
        'season_length': d['season_length'],
        'first_day': d['first_day'],
        'last_day': d['last_day']
    })
summary_df = pd.DataFrame(summary)
summary_df.to_csv('seasonal_event_summary.csv', index=False)

# Safe file names
def safe_filename(s):
    return s.replace('/', '-')

for key, d in results.items():
    d['events'].to_csv(f'{safe_filename(key)}_events.csv', index=False)

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
        intervals_df.to_csv(f'{safe_filename(key)}_intervals.csv', index=False)

# ========== Export intervals with censoring indicator (filtered only available groups) ==========
intervals_data = []
for key, d in results.items():
    events = d['events']
    intervals = d['intervals']
    if not intervals.empty and len(events) > 1:
        for idx, interval in enumerate(intervals):
            intervals_data.append({
                'group': key,
                'interval_days': interval.days,
                'censored': False
            })
    if len(events) == 0:
        intervals_data.append({
            'group': key,
            'interval_days': d['season_length'],
            'censored': True
        })
intervals_censor_df = pd.DataFrame(intervals_data)
intervals_censor_df.to_csv('intervals_with_censoring.csv', index=False)

# ========== Count events per group (use same available_groups logic!) ==========
event_counts = []
for dec in ['2005-2014', '2015-2025']:
    available_groups = df.loc[df['decade']==dec, ['season', 'season_group']].drop_duplicates()
    for _, (season, group_label) in available_groups.iterrows():
        subdf = df[(df['decade']==dec) & (df['season']==season) & (df['season_group']==group_label)]
        for label, thresh in [('heavy', 10), ('extreme', 20)]:
            events = find_events(subdf, threshold=thresh)
            event_counts.append({
                'decade': dec,
                'season': season,
                'season_group': group_label,
                'type': label,
                'n_events': len(events)
            })
event_counts_df = pd.DataFrame(event_counts)
event_counts_df.to_csv('yearly_seasonal_event_counts.csv', index=False)

print("\nAll results have been processed and exported.")

# ========== Visualization Example ==========
# Example: Plot all heavy winter intervals in 2005-2014
all_winter_intervals = []
for key, d in results.items():
    if key.startswith('heavy_2005-2014_Winter'):
        all_winter_intervals.extend(d['intervals'].dt.days.dropna().tolist())
all_winter_intervals = np.array(all_winter_intervals)

if len(all_winter_intervals) > 0:
    plt.hist(all_winter_intervals, bins=10, alpha=0.6, label='Intervals')
    x = np.linspace(0, max(all_winter_intervals), 100)
    exp_params = expon.fit(all_winter_intervals)
    plt.plot(x, len(all_winter_intervals)*(np.diff(np.histogram(all_winter_intervals, bins=10)[1])[0])
             *expon.pdf(x, *exp_params), label='Exp fit')
    if len(all_winter_intervals) > 1:
        try:
            gamma_params = gamma.fit(all_winter_intervals)
            plt.plot(x, len(all_winter_intervals)*(np.diff(np.histogram(all_winter_intervals, bins=10)[1])[0])
                     *gamma.pdf(x, *gamma_params), label='Gamma fit')
        except Exception as e:
            print(f"Gamma fit error: {e}")
    plt.legend()
    plt.title("All 2005-2014 Winter Heavy Event Intervals")
    plt.xlabel("Interval (days)")
    plt.ylabel("Count")
    plt.show()
else:
    print("No data for all winter intervals.")

all_winter_magnitudes = []
for key, d in results.items():
    if key.startswith('heavy_2005-2014_Winter'):
        all_winter_magnitudes.extend(d['magnitudes'].dropna().tolist())
all_winter_magnitudes = np.array(all_winter_magnitudes)

if len(all_winter_magnitudes) > 1:
    gev_params = fit_and_report(all_winter_magnitudes, 'gev')
    plt.hist(all_winter_magnitudes, bins=10, alpha=0.6, label='Event magnitude')
    x = np.linspace(min(all_winter_magnitudes), max(all_winter_magnitudes), 100)
    if gev_params is not None:
        c, loc, scale = gev_params
        plt.plot(x, len(all_winter_magnitudes)*(np.diff(np.histogram(all_winter_magnitudes, bins=10)[1])[0])*
                 genextreme.pdf(x, c, loc, scale), label='GEV fit')
    plt.legend()
    plt.title("All 2005-2014 Winter Heavy Event Magnitudes")
    plt.xlabel("Total precipitation (mm)")
    plt.ylabel("Count")
    plt.show()
else:
    print("Not enough magnitude data for GEV fitting and plotting.")
