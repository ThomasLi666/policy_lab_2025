import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, gamma, genextreme

# 1. Load and prepare the data 
file_path = 'C:/Users/123/Downloads/policy_lab/policy_lab_2025/event_and_magnitudes_analysis/historical_precipitation_fixed.csv'

df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Representative date'], dayfirst=True)
df['Precip'] = pd.to_numeric(df['Rainfall'], errors='coerce')
df = df[['Date', 'Precip']].dropna().sort_values('Date').reset_index(drop=True)
df['year'] = df['Date'].dt.year

# 2. Assign season and decade
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
df['decade'] = pd.cut(df['year'], bins=[2004, 2014, 2025],
                      labels=['2005-2014', '2015-2025'], right=True)

# 3. Define event detection function 
def find_events(subdf, threshold):
    events = []
    in_event = False
    start = None

    for i in subdf.index:
        val = subdf.at[i, 'Precip']
        if not in_event and val >= threshold:
            in_event = True
            start = i
        if in_event and (val < threshold or i == subdf.index[-1]):
            end = i if val >= threshold else i - 1
            event_rows = subdf.loc[start:end]
            events.append({
                'start_date': event_rows.iloc[0]['Date'],
                'end_date': event_rows.iloc[-1]['Date'],
                'duration': (event_rows.iloc[-1]['Date'] - event_rows.iloc[0]['Date']).days + 1,
                'total_precip': event_rows['Precip'].sum()
            })
            in_event = False
    return pd.DataFrame(events)

# 4. Analysis function for a single decade 
def analyze_decade(decade, season='Summer', threshold=10, interval_limit=120):
    subdf = df[(df['decade'] == decade) & (df['season'] == season)]
    events = find_events(subdf, threshold)

    if len(events) > 1:
        intervals = (
            events['start_date'].iloc[1:].reset_index(drop=True)
            - events['end_date'].iloc[:-1].reset_index(drop=True)
        )
        intervals_days = intervals.dt.days.dropna().values
    else:
        intervals_days = np.array([])

    magnitudes = events['total_precip'].dropna().values

    # Filter intervals below threshold
    valid_mask = intervals_days < interval_limit
    filtered_intervals = intervals_days[valid_mask]
    filtered_starts = events['start_date'].iloc[1:].reset_index(drop=True)[valid_mask]
    filtered_ends = events['end_date'].iloc[:-1].reset_index(drop=True)[valid_mask]

    # Export to CSV 
    label = decade.replace("-", "_") + f"_{season}"

    events.to_csv(f"{label}_events.csv", index=False)

    if len(filtered_intervals) > 0:
        pd.DataFrame({
            'previous_event_end_date': filtered_ends,
            'current_event_start_date': filtered_starts,
            'interval_days': filtered_intervals
        }).to_csv(f"{label}_intervals.csv", index=False)

    pd.DataFrame({
        'start_date': events['start_date'],
        'end_date': events['end_date'],
        'magnitude_mm': events['total_precip']
    }).to_csv(f"{label}_magnitudes.csv", index=False)

    # Print Summary
    print(f'\n====== {decade} {season} Heavy Events (≥{threshold}mm) ======')
    print(f'Total events: {len(events)}')
    print(f'Filtered intervals (<{interval_limit} days): {filtered_intervals}')
    print(f'Event magnitudes (mm): {magnitudes}')

    # Plot Interval Distribution
    if len(filtered_intervals) > 0:
        plt.figure(figsize=(8, 4))
        plt.hist(filtered_intervals, bins=10, alpha=0.6, label='Days')
        x = np.linspace(0, max(filtered_intervals), 100)
        exp_params = expon.fit(filtered_intervals)
        plt.plot(x, len(filtered_intervals) * np.diff(np.histogram(filtered_intervals, bins=10)[1])[0] *
                 expon.pdf(x, *exp_params), label='Exponential fit')
        try:
            gamma_params = gamma.fit(filtered_intervals)
            plt.plot(x, len(filtered_intervals) * np.diff(np.histogram(filtered_intervals, bins=10)[1])[0] *
                     gamma.pdf(x, *gamma_params), label='Gamma fit')
        except Exception as e:
            print(f"Gamma fit error: {e}")
        plt.title(
    f"{dec} Days between heavy Rainfall Event\n"
)
        plt.xlabel('days')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No valid intervals for fitting.")

    # Plot Magnitude Distribution 
    if len(magnitudes) > 1:
        plt.figure(figsize=(8, 4))
        gev_params = genextreme.fit(magnitudes)
        x = np.linspace(min(magnitudes), max(magnitudes), 100)
        plt.hist(magnitudes, bins=10, alpha=0.6, label='Observed magnitudes')
        plt.plot(x, len(magnitudes) * np.diff(np.histogram(magnitudes, bins=10)[1])[0] *
                 genextreme.pdf(x, *gev_params), label='GEV fit')
        plt.title(
    f"{dec} Summer Heavy Rainfall Event Magnitude\n"
    f"Magnitude = Total precipitation during each event")

        plt.xlabel('Total precipitation (mm)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough magnitude data for GEV fitting.")

# 5. Run for each decade 
for dec in ['2005-2014', '2015-2025']:
    analyze_decade(decade=dec, season='Summer', threshold=10)
