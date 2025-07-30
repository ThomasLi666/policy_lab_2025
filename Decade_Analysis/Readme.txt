This script performs decade-based analysis (2005–2014 and 2015–2025) for both heavy and extreme rainfall events.

The code generates statistics for:

The number of days (intervals) between heavy/extreme events

The rainfall accumulation (magnitude) of each heavy/extreme event

Event classification is as follows:

To avoid double-counting, a “heavy” rainfall event is defined as any event where the maximum daily precipitation is at least 10 mm but less than 20 mm.

Since an event may span multiple consecutive days, if any day within an event records precipitation of 20 mm or more, the entire event is classified as an “extreme” event rather than a “heavy” event.

This ensures that each event is counted only once and is assigned to the appropriate category based on its maximum daily precipitation.