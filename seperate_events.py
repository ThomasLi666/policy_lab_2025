import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the data
csv_path = 'C:/Users/123/Downloads/policy_lab/historical_precipitation_fixed.csv'
df = pd.read_csv(csv_path, dtype=str)
df = df[df['Representativt dygn'].str.match(r'\d{4}-\d{2}-\d{2}', na=False)].copy()
df['date'] = pd.to_datetime(df['Representativt dygn'])
df['precip'] = df['Nederbördsmängd'].str.replace(',', '.').astype(float)

# 2. Set thresholds
THRESHOLD_HEAVY = 10
THRESHOLD_EXTREME = 20

def find_multi_day_events(bool_series, min_length=2):
    """Find intervals where True lasts for at least min_length days."""
    in_event = False
    events = []
    start = None
    for i, val in enumerate(bool_series):
        if val and not in_event:
            in_event = True
            start = i
        if (not val or i == len(bool_series)-1) and in_event:
            end = i if not val else i+1
            if end - start >= min_length:
                events.append((start, end))
            in_event = False
    return events

# 3. Find heavy rain events (>=10mm, at least 2 consecutive days)
is_heavy = df['precip'] >= THRESHOLD_HEAVY
heavy_events = find_multi_day_events(is_heavy.values, min_length=2)

# 4. Find extreme rain events (>=20mm, at least 2 consecutive days)
is_extreme = df['precip'] >= THRESHOLD_EXTREME
extreme_events = find_multi_day_events(is_extreme.values, min_length=2)

# 5. Plot the results
plt.figure(figsize=(14,6))
plt.plot(df['date'], df['precip'], color='lightgrey', lw=1, label='Precipitation')

heavy_label_shown = False
extreme_label_shown = False

# Highlight heavy rain events in green
for (start, end) in heavy_events:
    plt.bar(df['date'].iloc[start:end], df['precip'].iloc[start:end],
            width=1, color='green', label='Heavy Event (>=10mm, >=2 days)' if not heavy_label_shown else "")
    heavy_label_shown = True

# Highlight extreme rain events in red
for (start, end) in extreme_events:
    plt.bar(df['date'].iloc[start:end], df['precip'].iloc[start:end],
            width=1, color='red', label='Extreme Event (>=20mm, >=2 days)' if not extreme_label_shown else "")
    extreme_label_shown = True

plt.axhline(THRESHOLD_HEAVY, color='green', ls='--', lw=1.5, label='Heavy Threshold (10mm)')
plt.axhline(THRESHOLD_EXTREME, color='red', ls='--', lw=1.5, label='Extreme Threshold (20mm)')
plt.xlabel('Date')
plt.ylabel('Daily Precipitation (mm)')
plt.title('Heavy & Extreme Rain Events (Bar Plot, Continuous >=2 days)')
plt.legend()
plt.tight_layout()
plt.show()

# 6. Print statistics
print(f'Heavy rain events (>=10mm, >=2 consecutive days): {len(heavy_events)}')
print(f'Extreme rain events (>=20mm, >=2 consecutive days): {len(extreme_events)}')

# Export detailed event information
def export_events(events, name):
    results = []
    for (start, end) in events:
        event_df = df.iloc[start:end]
        results.append({
            'start_date': event_df['date'].iloc[0],
            'end_date': event_df['date'].iloc[-1],
            'duration_days': end - start,
            'total_precip': event_df['precip'].sum(),
            'max_precip': event_df['precip'].max()
        })
    outdf = pd.DataFrame(results)
    outdf.to_csv(f'{name}_rain_events.csv', index=False)
    return outdf

heavy_df = export_events(heavy_events, 'heavy')
extreme_df = export_events(extreme_events, 'extreme')

print(' CSV files for heavy and extreme events have been exported.')
