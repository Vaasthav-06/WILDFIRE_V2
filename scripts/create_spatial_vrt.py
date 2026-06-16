"""
scripts/create_spatial_vrt.py

Builds a Virtual Raster Tile (.vrt) index mapping file from the 17 fragmented
ESA land cover tiles using a clean XML layout standard. Requires zero native GDAL.
"""

from pathlib import Path
import rasterio
from loguru import logger

from src.config import DATA_RAW

logger.add("logs/create_vrt.log", rotation="5 MB")

def main():
    exports_dir = DATA_RAW / "Wildfire_Data_Exports"
    output_vrt = DATA_RAW / "esa_landcover_india.vrt"
    
    logger.info(f"Scanning for raster fragments in: {exports_dir}")
    
    # Collect all 17 fragmented tiles matching your file pattern
    tiff_files = sorted([p for p in exports_dir.glob("esa_landcover_india-*.tif")])
    
    if not tiff_files:
        raise FileNotFoundError(
            f"No raster fragments found in {exports_dir}. Ensure your download completed."
        )
        
    logger.info(f"Found {len(tiff_files)} fragmented tiles. Extracting master grid coordinates...")
    
    # Read spatial envelope geometry attributes from the very first block tile
    with rasterio.open(tiff_files[0]) as first_tile:
        crs_wkt = first_tile.crs.wkt
        width = first_tile.width
        height = first_tile.height
        transform = first_tile.transform
        raw_dtype = first_tile.dtypes[0]
        
        # Translate numpy dtypes to strict GDAL VRT standards
        gdal_dtype_map = {
            "uint8": "Byte", "int8": "Byte",
            "uint16": "UInt16", "int16": "Int16",
            "uint32": "UInt32", "int32": "Int32",
            "float32": "Float32", "float64": "Float64"
        }
        dtype = gdal_dtype_map.get(raw_dtype, "Float32")

    logger.info("Assembling virtual layout string data blocks...")
    
    # Construct a standardized spatial VRT structure mapping XML configuration block
    vrt_xml = f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
  <SRS dataAxisToSRSAxisMapping="2,1">{crs_wkt}</SRS>
  <GeoTransform>{transform.c}, {transform.a}, {transform.b}, {transform.f}, {transform.d}, {transform.e}</GeoTransform>
  <VRTRasterBand dataType="{dtype}" band="1">
"""
    
    # Map the relative spatial orientation for each underlying file tile structure
    for idx, filepath in enumerate(tiff_files, start=1):
        with rasterio.open(filepath) as src:
            t = src.transform
            vrt_xml += f"""    <ComplexSource>
      <SourceFilename relativeToVRT="1">Wildfire_Data_Exports/{filepath.name}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{src.width}" ySize="{src.height}" />
      <DstRect xOff="0" yOff="0" xSize="{src.width}" ySize="{src.height}" />
    </ComplexSource>
"""
            
    vrt_xml += """  </VRTRasterBand>
</VRTDataset>"""

    logger.info("Writing virtual spatial index map file manifest data to host machine...")
    with open(output_vrt, "w", encoding="utf-8") as f:
        f.write(vrt_xml.strip())
        
    if output_vrt.exists() and output_vrt.stat().st_size > 0:
        logger.info(f"Success! Virtual spatial index map file written perfectly at: {output_vrt}")
    else:
        logger.error("VRT manifest configuration save failed.")

if __name__ == "__main__":
    main()