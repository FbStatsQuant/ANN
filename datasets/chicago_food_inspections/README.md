# Chicago Food Inspections (not committed, 340 MB)

- Source: https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5
- Binary target: collapse `Results` into Pass (Pass, Pass w/ Conditions) vs. Fail.
- The `Violations` column is long free text, which is why the file is so large. Drop it or select columns to shrink it.

Download:

    curl -L "https://data.cityofchicago.org/resource/4ijn-s7e5.csv?\$limit=300000" -o food_inspections.csv

Smaller version without the text column:

    curl -L "https://data.cityofchicago.org/resource/4ijn-s7e5.csv?\$limit=300000&\$select=inspection_id,dba_name,facility_type,risk,zip,inspection_date,inspection_type,results,latitude,longitude" -o food_inspections.csv
