import os
import urllib.request
import ssl

# Define paths matching your project pipeline
output_dir = r"C:\Users\vaast\OneDrive\Desktop\wildfire_v2\data\raw"
output_file = os.path.join(output_dir, "srtm_india.tif")

os.makedirs(output_dir, exist_ok=True)

# Direct open-access URL to a pre-compiled, optimized country-scale DEM raster for India 
# (Bypasses grid-slicing completely to prevent tile-limit errors)
url = "https://github.com/vincentsarago/asf-dem-alpha/raw/main/data/IND_wgs84.tif"

print("Connecting to the optimized open-access geospatial asset repository...")
print("Downloading country-wide India Elevation Map directly (No tile splitting)...")

try:
    # Bypass local SSL certificate strict verification issues if any exist on the network
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    # Download the single-raster asset directly to your raw data path
    with urllib.request.urlopen(url, context=ctx) as response, open(output_file, 'wb') as out_file:
        out_file.write(response.read())
        
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"\nSuccess! Your elevation asset is ready at: {output_file}")
        print("Verified file path size looks strong. Ready for mapping extraction pipeline.")
    else:
        print("\nDownload complete, but the generated file appears empty. Please check connection.")
        
except Exception as e:
    print(f"\nAn error occurred during file fetch: {e}")