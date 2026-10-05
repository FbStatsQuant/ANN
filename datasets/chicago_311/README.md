# Chicago 311 Service Requests 2024 (sample) - multiclass

**The data file is NOT on GitHub (59 MB). It lives on the local computer** at
`datasets/chicago_311/service_requests_2024_sample.csv` and is gitignored. To rebuild it on another machine:

    python datasets/fetch_chicago.py

- Source: https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy
- Sample: every request on the 1st, 8th, 15th and 22nd of each month of 2024 (48 full days, 261,269 requests).
- Target: `owner_department` (14 departments, ticket routing) or `sr_type` (105 request types, long tail).
- Leakage: `status` and `closed_date` are only known after the request is handled. `sr_short_code` encodes `sr_type`.
