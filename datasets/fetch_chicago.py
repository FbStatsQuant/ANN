# Download helper: fetches whole sampled days from the Chicago Data Portal, then concatenates.
import os, time, requests, pandas as pd
JOBS = [
    ("wrvz-psew", "trip_start_timestamp", 2023, [1, 15], "chicago_taxi", "taxi_trips_2023_sample.csv",
     "trip_id,taxi_id,trip_start_timestamp,trip_end_timestamp,trip_seconds,trip_miles,pickup_community_area,dropoff_community_area,fare,tips,tolls,extras,trip_total,payment_type,company,pickup_centroid_latitude,pickup_centroid_longitude,dropoff_centroid_latitude,dropoff_centroid_longitude"),
    ("v6vf-nfxy", "created_date", 2024, [1, 8, 15, 22], "chicago_311", "service_requests_2024_sample.csv",
     "sr_number,sr_type,sr_short_code,owner_department,status,origin,created_date,closed_date,street_address,zip_code,ward,community_area,police_district,latitude,longitude,duplicate,legacy_record"),
]
ROOT = os.path.dirname(os.path.abspath(__file__))  # the datasets/ folder
for ds, col, year, days, folder, out, cols in JOBS:
    parts = os.path.join(ROOT, folder, "_parts"); os.makedirs(parts, exist_ok=True)
    for m in range(1, 13):
        for d in days:
            p = os.path.join(parts, f"{year}-{m:02d}-{d:02d}.csv")
            if os.path.exists(p): continue
            q = {"$select": cols, "$where": f"{col} between '{year}-{m:02d}-{d:02d}T00:00:00' and '{year}-{m:02d}-{d:02d}T23:59:59'",
                 "$order": col, "$limit": 200000}
            for attempt in range(5):
                try:
                    r = requests.get(f"https://data.cityofchicago.org/resource/{ds}.csv", params=q, timeout=300)
                    r.raise_for_status(); open(p + ".tmp", "wb").write(r.content); os.replace(p + ".tmp", p)
                    print(folder, p[-14:], len(r.content) // 1024, "KB", flush=True); break
                except Exception as e:
                    print("retry", folder, m, d, e, flush=True); time.sleep(10)
    frames = [pd.read_csv(os.path.join(parts, f), low_memory=False) for f in sorted(os.listdir(parts)) if f.endswith(".csv")]
    df = pd.concat(frames, ignore_index=True); df.to_csv(os.path.join(ROOT, folder, out), index=False)
    print("DONE", folder, df.shape, len(frames), "days", flush=True)
# Usage: python datasets/fetch_chicago.py  (resumable; delete datasets/*/_parts when it finishes)
