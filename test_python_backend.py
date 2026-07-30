import urllib.request
import json

url = "http://localhost:8000/predict/24h"
data = {
    "area": None,
    "include_spread": True,
    "include_explanation": True,
    "min_prob_geojson": 0.15
}
req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})

print("Sending request to python backend...")
try:
    with urllib.request.urlopen(req, timeout=10) as response:
        print("Status code:", response.getcode())
        print("Response length:", len(response.read()))
except Exception as e:
    print("Error:", e)
