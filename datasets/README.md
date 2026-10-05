# Datasets

| Folder | Task | Size | On GitHub? |
|---|---|---|---|
| [bank_marketing](bank_marketing/) | binary | 41k rows | yes |
| [credit_card_default](credit_card_default/) | binary | 30k rows | yes |
| [bike_sharing](bike_sharing/) | regression | 17k rows (hourly) | yes |
| [chicago_taxi](chicago_taxi/) | regression | 433k rows, 147 MB | no, local only |
| [dry_bean](dry_bean/) | multiclass | 13.6k rows | yes |
| [chicago_311](chicago_311/) | multiclass | 261k rows, 59 MB | no, local only |

All data lives on the local computer. The two Chicago files are too heavy for GitHub and are gitignored;
`fetch_chicago.py` rebuilds them (see each folder's README).
