import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, gamma, genextreme
import os

# 1. Load and prepare the data
# Read the precipitation data and parse relevant columns
file_path = 'C:/Users/123/Downloads/policy_lab/policy_lab_2025/event_and_magnitudes_analysis/historical_precipitation_fixed.csv'

df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Representative date'], dayfirst=True)
df['Precip'] = pd.to_numeric(df['Rainfall'], errors='coerce')
df = df[['Date', 'Precip']].dropna().sort_values('Date').reset_index(drop=True)
df['year'] = df['Date'].dt.year

# 2. Assign season and decade
def get_season(month):
    """Classify a month into its corresponding season."""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

df['season'] = df['Date'].dt.month.apply(get_season)
df['decade'] = pd.cut(
    df['year'], bins=[2004, 2014, 2025],
    labels=['2005-2014', '2015-2025'], right=True
)

# 3. Event detection and classification
def find_events_with_classification(subdf, heavy_thr=10, extreme_thr=20):
    """
    Identify all events with daily precipitation >= heavy_thr.
    Classify an event as 'extreme' if any day in the event has precipitation >= extreme_thr,
    otherwise classify as 'heavy'.
    """
    events = []
    in_event = False
    start = None

    for i in subdf.index:
        val = subdf.at[i, 'Precip']
        if not in_event and val >= heavy_thr:
            in_event = True
            start = i
        if in_event and (val < heavy_thr or i == subdf.index[-1]):
            end = i if val >= heavy_thr else i - 1
            event_rows = subdf.loc[start:end]
            # Classify the event type
            if (event_rows['Precip'] >= extreme_thr).any():
                event_type = 'extreme'
            else:
                event_type = 'heavy'
            events.append({
                'start_date': event_rows.iloc[0]['Date'],
                'end_date': event_rows.iloc[-1]['Date'],
                'duration': (event_rows.iloc[-1]['Date'] - event_rows.iloc[0]['Date']).days + 1,
                'total_precip': event_rows['Precip'].sum(),
                'event_type': event_type
            })
            in_event = False
    return pd.DataFrame(events)

# 4. Analysis and plotting for a single decade and event type
def analyze_decade(decade, season='Summer', interval_limit=120, event_type_select='heavy'):
    """
    Analyze all events of the selected type (heavy/extreme) for a given decade and season.
    Outputs summary statistics, CSV files, and distribution plots.
    """
    subdf = df[(df['decade'] == decade) & (df['season'] == season)]
    events = find_events_with_classification(subdf, heavy_thr=10, extreme_thr=20)
    events = events[events['event_type'] == event_type_select]

    # Calculate intervals between events
    if len(events) > 1:
        intervals = (
            events['start_date'].iloc[1:].reset_index(drop=True)
            - events['end_date'].iloc[:-1].reset_index(drop=True)
        )
        intervals_days = intervals.dt.days.dropna().values
    else:
        intervals_days = np.array([])

    magnitudes = events['total_precip'].dropna().values

    # Filter intervals: only keep those < interval_limit days (to exclude across-season/year gaps)
    valid_mask = intervals_days < interval_limit
    filtered_intervals = intervals_days[valid_mask]
    filtered_starts = events['start_date'].iloc[1:].reset_index(drop=True)[valid_mask]
    filtered_ends = events['end_date'].iloc[:-1].reset_index(drop=True)[valid_mask]

    # Export results to CSV files for reproducibility and further analysis
    label = f"{decade.replace('-', '_')}_{season}_{event_type_select.capitalize()}"
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

    # Print basic summary to console
    type_label = 'Heavy (10mm≤max<20mm)' if event_type_select=='heavy' else 'Extreme (≥20mm)'
    print(f'\n====== {decade} {season} {type_label} Events ======')
    print(f'Total events: {len(events)}')
    print(f'Filtered intervals (<{interval_limit} days): {filtered_intervals}')
    print(f'Event magnitudes (mm): {magnitudes}')

    # Plot interval distribution and fit statistical models
    if len(filtered_intervals) > 0:
        plt.figure(figsize=(8, 4))
        plt.hist(filtered_intervals, bins=10, alpha=0.6, label='Intervals (days)')
        x = np.linspace(0, max(filtered_intervals), 100)
        exp_params = expon.fit(filtered_intervals)
        plt.plot(
            x,
            len(filtered_intervals) * np.diff(np.histogram(filtered_intervals, bins=10)[1])[0] *
            expon.pdf(x, *exp_params),
            label='Exponential fit'
        )
        try:
            gamma_params = gamma.fit(filtered_intervals)
            plt.plot(
                x,
                len(filtered_intervals) * np.diff(np.histogram(filtered_intervals, bins=10)[1])[0] *
                gamma.pdf(x, *gamma_params),
                label='Gamma fit'
            )
        except Exception as e:
            print(f"Gamma fit error: {e}")
        plt.title(
            f" Days between {type_label} Events in all {season} in {decade} \n"
        
        )
        plt.xlabel('Days between events')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{label}_intervals.png", dpi=300)
        plt.show()
    else:
        print("No valid intervals for fitting.")

    # Plot magnitude (total precipitation) distribution and fit GEV
    if len(magnitudes) > 1:
        plt.figure(figsize=(8, 4))
        gev_params = genextreme.fit(magnitudes)
        x = np.linspace(min(magnitudes), max(magnitudes), 100)
        plt.hist(magnitudes, bins=10, alpha=0.6, label='Event magnitudes')
        plt.plot(
            x,
            len(magnitudes) * np.diff(np.histogram(magnitudes, bins=10)[1])[0] *
            genextreme.pdf(x, *gev_params),
            label='GEV fit'
        )
        plt.title(
            f"{type_label} rainfall Event---Accumulation in all  {season} in {decade} \n"
           
        )
        plt.xlabel('Total precipitation (mm)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{label}_magnitudes.png", dpi=300)
        plt.show()
    else:
        print("Not enough magnitude data for GEV fitting.")

# 5. Run for each decade and both event types
for dec in ['2005-2014', '2015-2025']:
    # Analyze both heavy and extreme events for each decade
    analyze_decade(decade=dec, season='Summer', interval_limit=120, event_type_select='heavy')
    analyze_decade(decade=dec, season='Summer', interval_limit=120, event_type_select='extreme')
