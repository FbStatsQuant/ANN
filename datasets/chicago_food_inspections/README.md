# Chicago Food Inspections

**The data file is NOT in this repo (345 MB). It lives only on the local computer** at
`datasets/chicago_food_inspections/food_inspections.csv` and is gitignored. If you cloned this repo elsewhere, download it first:

    curl -L 'https://data.cityofchicago.org/resource/4ijn-s7e5.csv?$limit=500000' -o food_inspections.csv

- Source: https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5
- Binary target: collapse `results` into Pass (Pass, Pass w/ Conditions) vs. Fail.
- The `violations` column is long free text and is why the file is so large.

