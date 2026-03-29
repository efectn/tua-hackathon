#!/bin/bash

# Get the directory where the script is located (root folder)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Fetching data.json (Range: 2023-09-25 to 2024-12-15) for the backend..."
# Note: This is an extremely long time range (~15 months) of 1-minute data! It may take a couple minutes to download.
curl -o "$DIR/backend/data.json" "https://cdaweb.gsfc.nasa.gov/hapi/data?id=OMNI_HRO_1MIN&parameters=F,BZ_GSM,flow_speed,proton_density,T,E,SYM_H&time.min=2023-09-25T00:00:00Z&time.max=2024-12-15T00:00:00Z&format=json"

echo ""
echo "Fetching data_test.json (Range: 2024-05-10 to 2024-05-15) for testing..."
curl -o "$DIR/data_test.json" "https://cdaweb.gsfc.nasa.gov/hapi/data?id=OMNI_HRO_1MIN&parameters=F,BZ_GSM,flow_speed,proton_density,T,E,SYM_H&time.min=2024-05-10T00:00:00Z&time.max=2024-05-15T00:00:00Z&format=json"

echo ""
echo "Download complete!"
