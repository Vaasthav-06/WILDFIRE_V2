import json
from pathlib import Path

# Ensure directory exists
Path("data/raw").mkdir(parents=True, exist_ok=True)

bbox = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [68.1, 8.0],
                [97.4, 8.0],
                [97.4, 37.6],
                [68.1, 37.6],
                [68.1, 8.0],
            ]]
        },
        "properties": {"name": "India"}
    }]
}

with open("data/raw/india_bbox.geojson", "w") as f:
    json.dump(bbox, f)
print("Saved data/raw/india_bbox.geojson")