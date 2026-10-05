# Bike Sharing (UCI) - regression

- Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- Files: `hour.csv` (17,379 rows, hourly) and `day.csv` (731 rows, daily), Capital Bikeshare in Washington DC, 2011 to 2012.
- Target: `cnt` (total rentals). `casual + registered = cnt`, so drop both as features (leakage).
- Column descriptions are in `description.txt`.
