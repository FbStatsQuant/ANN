# Chicago Taxi Trips 2023 (sample) - regression

**The data file is NOT on GitHub (147 MB). It lives on the local computer** at
`datasets/chicago_taxi/taxi_trips_2023_sample.csv` and is gitignored. To rebuild it on another machine:

    python datasets/fetch_chicago.py

- Source: https://data.cityofchicago.org/Transportation/Taxi-Trips-2013-2023-/wrvz-psew
- Sample: every trip on the 1st and 15th of each month of 2023 (24 full days, 433,009 trips), so seasonality and within-day patterns are preserved.
- Target: `fare` (or `trip_seconds` for an ETA model). Raw data: expect zero-mile trips, missing locations and extreme fares.
- Leakage: `trip_total`, `tips`, `tolls`, `extras` and `trip_end_timestamp` are only known after the trip.
