import os
import requests
from datetime import datetime, timedelta, timezone

HAPI_BASE = "https://cdaweb.gsfc.nasa.gov/hapi/data"
DATASET_ID = "OMNI_HRO_1MIN"
PARAMETERS = "F,BZ_GSM,flow_speed,proton_density,T,E,SYM_H"

def fetch_data_to_json(filepath, days=3):
    # OMNI data is delayed by ~2 months, so fetch 65 days ago
    end = datetime.now(timezone.utc) - timedelta(days=65)
    start = end - timedelta(days=days)
    
    params = {
        "id": DATASET_ID,
        "parameters": PARAMETERS,
        "time.min": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time.max": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "format": "json"
    }
    
    print(f"Fetching data to {filepath} ...")
    try:
        resp = requests.get(HAPI_BASE, params=params, timeout=30)
        resp.raise_for_status()
        with open(filepath, "w") as f:
            f.write(resp.text)
        print(f"Successfully saved {filepath}\n")
    except Exception as e:
        print(f"Failed to fetch data: {str(e)}\n")

if __name__ == "__main__":
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(backend_dir)
    
    data_json_path = os.path.join(backend_dir, "data.json")
    data_test_path = os.path.join(root_dir, "data_test.json")
    
    fetch_data_to_json(data_json_path, days=3)
    fetch_data_to_json(data_test_path, days=3)

    print("Done!")
