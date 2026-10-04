# Chicago Crimes (2023)

- Source: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2
- 263,851 rows, 14 columns, saved as `crimes_2023.csv.gz` (pandas reads it directly).
- Binary target: `arrest` (about 12% true, so imbalanced).
- Multiclass target: `primary_type`. Regression idea: daily crime counts per district.

Re-download (other years: change `year=2023`):

    curl -L "https://data.cityofchicago.org/resource/ijzp-q8t2.csv?\$where=year=2023&\$limit=500000" -o crimes.csv
