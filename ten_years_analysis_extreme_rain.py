import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, gamma, genextreme
import os

# ========== 1. Read and clean data ==========
df = pd.read_csv('C:/Users/123/Downloads/policy_lab/policy_lab_2025/event_and_magnitudes_analysis/historical_precipitation_fixed.csv')
df['Date'] = pd.to_datetime(df['Representative date'], dayfirst=True)
df['Precip'] = pd.to_numeric(df['Rainfall'], errors='coerce')
df = df[['Date', 'Precip']].sort_values('Date').reset_index(drop=True)
df['year'] = df['Date'].dt.year

# ========== 2. Assign season and decade ==========
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
df['decade'] = pd.cut(
    df['year'],
    bins=[2004, 2014, 2025],
    labels=['2005-2014', '2015-2025'],
    right=True
)

# ========== 3. Define event detection function ==========
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

# ========== 4. Main analysis loop (Extreme Events ≥20mm) ==========
threshold = 20
season = 'Summer'
decades = ['2005-2014', '2015-2025']

for dec in decades:
    # Subset data for given decade and season
    subdf = df[(df['decade'] == dec) & (df['season'] == season)]
    events = find_events(subdf, threshold)

    # Compute intervals between events
    if len(events) > 1:
        intervals = (
            events['start_date'].iloc[1:].reset_index(drop=True)
            - events['end_date'].iloc[:-1].reset_index(drop=True)
        )
        intervals_days_all = intervals.dt.days.dropna().values
    else:
        intervals_days_all = np.array([])

    magnitudes = events['total_precip'].dropna().values if len(events) > 0 else np.array([])

    # Filter intervals < 120 days
    valid_mask = intervals_days_all < 120
    filtered_intervals_days = intervals_days_all[valid_mask]
    filtered_interval_starts = events['start_date'].iloc[1:].reset_index(drop=True)[valid_mask]
    filtered_interval_ends = events['end_date'].iloc[:-1].reset_index(drop=True)[valid_mask]

    # ========== 5. Print summary ==========
    print(f'\n====== {dec} {season} Extreme Events (≥{threshold}mm) ======')
    print(f'Number of events: {len(events)}')
    print(f'Filtered intervals (<120 days): {filtered_intervals_days}')
    print(f'Magnitudes (mm): {magnitudes}')

    # ========== 6. Export results to CSV ==========
    label = dec.replace("-", "_") + f"_{season}_Extreme"
    
    events.to_csv(f"{label}_events.csv", index=False)

    if len(filtered_intervals_days) > 0:
        pd.DataFrame({
            'previous_event_end_date': filtered_interval_ends,
            'current_event_start_date': filtered_interval_starts,
            'interval_days': filtered_intervals_days
        }).to_csv(f"{label}_intervals.csv", index=False)

    if len(events) > 0:
        events[['start_date', 'end_date', 'total_precip']].rename(
            columns={'total_precip': 'magnitude_mm'}
        ).to_csv(f"{label}_magnitudes.csv", index=False)

    # 7. Plot event intervals 
    plt.figure(figsize=(8, 4))
    if len(filtered_intervals_days) > 0:
        plt.hist(filtered_intervals_days, bins=10, alpha=0.6, label='Days')
        x = np.linspace(0, max(filtered_intervals_days), 100)
        
        # Exponential fit
        exp_params = expon.fit(filtered_intervals_days)
        plt.plot(
            x,
            len(filtered_intervals_days) * (np.diff(np.histogram(filtered_intervals_days, bins=10)[1])[0]) *
            expon.pdf(x, *exp_params),
            label='Exponential fit'
        )

        # Gamma fit
        if len(filtered_intervals_days) > 1:
            try:
                gamma_params = gamma.fit(filtered_intervals_days)
                plt.plot(
                    x,
                    len(filtered_intervals_days) * (np.diff(np.histogram(filtered_intervals_days, bins=10)[1])[0]) *
                    gamma.pdf(x, *gamma_params),
                    label='Gamma fit'
                )
            except Exception as e:
                print(f"Gamma fit error: {e}")
        plt.title(
    f"{dec} Days between Extreme Rainfall Event\n"
)
        plt.xlabel('days')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No interval data available for fitting.")

    # 8. Plot event magnitudes 
    plt.figure(figsize=(8, 4))
    if len(magnitudes) > 1:
        gev_params = genextreme.fit(magnitudes)
        plt.hist(magnitudes, bins=10, alpha=0.6, label='Event magnitude')
        x = np.linspace(min(magnitudes), max(magnitudes), 100)
        plt.plot(
            x,
            len(magnitudes) * (np.diff(np.histogram(magnitudes, bins=10)[1])[0]) *
            genextreme.pdf(x, *gev_params),
            label='GEV fit'
        )
        plt.title(
    f"{dec} Summer Extreme Rainfall Event Magnitudes \n"
    f"Magnitude = Total precipitation during each event"
)

        plt.xlabel('Total precipitation (mm)')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough magnitude data for GEV fitting and plotting.")
