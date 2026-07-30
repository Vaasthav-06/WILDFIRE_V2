import urllib.request
import json
import time

url = "http://localhost:8000/predict/24h"
data = {
    "area": {
      "lat_min": 8.0,
      "lat_max": 37.6,
      "lon_min": 68.1,
      "lon_max": 97.4,
      "resolution": 0.2
    },
    "include_spread": True,
    "include_explanation": False,
    "min_prob_geojson": 0.15
}
req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})

print("Sending request to python backend without explanations...")
t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=30) as response:
        print("Status code:", response.getcode())
        resp_data = response.read()
        print("Response length:", len(resp_data))
        print(f"Elapsed: {time.time() - t0:.2f}s")
except Exception as e:
    print("Error:", e)
