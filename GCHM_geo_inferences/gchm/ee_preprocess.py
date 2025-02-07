import ee
import json
import os
from datetime import datetime, timedelta
import time
import numpy as np
import requests
from io import BytesIO

def initialize_gee(service_account_key):
    """Authenticate and initialize Google Earth Engine."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key
    ee.Initialize()

def load_and_validate_geojson(geojson_path):
    """Load and validate the GeoJSON file."""
    with open(geojson_path, 'r') as geojson_file:
        geojson = json.load(geojson_file)
    return ee.Geometry.Polygon(geojson['features'][0]['geometry']['coordinates'])

import json
import ee

def load_and_validate_geojson(geojson_path):
    """Load and validate the GeoJSON file."""
    with open(geojson_path, 'r') as geojson_file:
        geojson = json.load(geojson_file)
    return ee.Geometry.Polygon(geojson['features'][0]['geometry']['coordinates'])

def get_square_encompassing_polygon(polygon):
    """Generate a square that fully encompasses the given polygon."""
    bounds = polygon.bounds().coordinates().getInfo()[0]
    
    min_x = min([p[0] for p in bounds])
    max_x = max([p[0] for p in bounds])
    min_y = min([p[1] for p in bounds])
    max_y = max([p[1] for p in bounds])

    width = max_x - min_x
    height = max_y - min_y
    side = max(width, height)

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    square_coords = [
        [center_x - side / 2, center_y - side / 2],
        [center_x + side / 2, center_y - side / 2],
        [center_x + side / 2, center_y + side / 2],
        [center_x - side / 2, center_y + side / 2],
        [center_x - side / 2, center_y - side / 2]  # close the polygon
    ]

    return ee.Geometry.Polygon([square_coords])



def create_composite(start_date, polygon, cloud_percentage=50):
    """Create a composite with lower cloud coverage."""
    start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = start_date_obj + timedelta(days=90)
    end_date = end_date_obj.strftime("%Y-%m-%d")
    
    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(polygon)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_percentage)))
    
    sorted_collection = collection.sort('CLOUDY_PIXEL_PERCENTAGE')
    lowest_cloud_images = sorted_collection.limit(3)
    
    return lowest_cloud_images.median()



def export_image_to_drive(image, file_name, output_dir, polygon):
    """Export the image to Google Drive and wait until it's done."""
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=file_name,
        folder=output_dir,
        scale=10,
        region=polygon,
        crs='EPSG:4326',
        fileFormat='GeoTIFF',
        maxPixels=1e9
    )
    task.start()

    # Wait for the task to finish
    while task.active():
        print(f"Exporting {file_name}...")  # Optional: Display the export status
        time.sleep(30)  # Sleep for 10 seconds before checking again
    
    print(f"Export of {file_name} completed.")


# Function to get the DLC mask from Dynamic World (GOOGLE/DYNAMICWORLD/V1)
import ee
import numpy as np
import requests
from io import BytesIO
from datetime import datetime, timedelta

# Initialize Earth Engine
ee.Initialize()

def get_dlc_mask(aoi, date_array):
    """
    For each date in date_array, tries to get the mask from Dynamic World.
    If no DW mask is generated (or if it is empty), we use a static mask from ESA WorldCover.
    The returned mask is a NumPy array (unstructured).

    Parameters:
      - aoi: area of interest (an ee.Geometry)
      - date_array: list of dates (format "YYYY-MM-DD")

    Returns:
      - Dictionary mapping each date to a NumPy array representing the mask
    """

    masks = {}  # Dictionary to store masks for each date

    # --- Load ESA WorldCover mask once ---
    try:
        # Load ESA WorldCover v200 (this is a collection, so we take the first image)
        esa_image = ee.ImageCollection("ESA/WorldCover/v200").first().clip(aoi)
        # Define excluded classes for ESA (e.g., 50, 60, 70, 80)
        exclude_classes_esa = [50, 60, 70, 80]

        # Create a binary mask: pixel value is 1 if "Map" is not in excluded classes
        mask_product = ee.Image(1)
        for cls in exclude_classes_esa:
            mask_product = mask_product.multiply(esa_image.select("Map").neq(cls))

        esa_mask_resampled = mask_product.reproject(crs='EPSG:4326', scale=10)

        # Get the download URL for the ESA mask in NPY format
        url_esa = esa_mask_resampled.getDownloadURL({
            'scale': 10,
            'region': aoi,
            'format': 'NPY'
        })
        response_esa = requests.get(url_esa, stream=True)
        if response_esa.status_code == 200:
            mask_array_esa = np.load(BytesIO(response_esa.content))
            if mask_array_esa.dtype.names is not None:
                field = mask_array_esa.dtype.names[0]
                esa_mask_numeric = mask_array_esa[field]
            else:
                esa_mask_numeric = mask_array_esa
            print("ESA mask successfully loaded.")
        else:
            print("HTTP error", response_esa.status_code, "while downloading ESA mask.")
            esa_mask_numeric = None
    except Exception as e:
        print("Exception while loading ESA mask:", e)
        esa_mask_numeric = None

    # --- Process each date ---
    for start_date in date_array:
        mask_numeric = None  # Final mask for this date
        # Compute the 90-day period from start_date
        try:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_obj = start_date_obj + timedelta(days=90)
            end_date = end_date_obj.strftime("%Y-%m-%d")
        except Exception as e:
            print(f"[{start_date}] Error computing date range: {e}")
            continue

        # 1. Attempt to get Dynamic World mask
        try:
            dlc_collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            dlc_filtered = dlc_collection.filterBounds(aoi).filterDate(start_date, end_date).first()
            if dlc_filtered is not None:
                dlc_filtered = dlc_filtered.clip(aoi)
                # Define excluded classes for DW (e.g., 0, 5, 6, 7, 8)
                exclude_classes = [0, 5, 6, 7, 8]
                # Create a binary mask: pixel is 1 if it does not belong to excluded classes
                dlc_mask = dlc_filtered.select("label").neq(ee.Image.constant(exclude_classes)).reduce(ee.Reducer.min())
                dlc_mask_binary = dlc_mask.eq(1)
                dlc_mask_resampled = dlc_mask_binary.reproject(crs='EPSG:4326', scale=10)

                url_dw = dlc_mask_resampled.getDownloadURL({
                    'scale': 10,
                    'region': aoi,
                    'format': 'NPY'
                })
                response_dw = requests.get(url_dw, stream=True)
                if response_dw.status_code == 200:
                    mask_array = np.load(BytesIO(response_dw.content))
                    if mask_array.dtype.names is not None:
                        field = mask_array.dtype.names[0]
                        mask_numeric = mask_array[field]
                    else:
                        mask_numeric = mask_array
                    print(f"[{start_date}] DW mask generated.")
                else:
                    print(f"[{start_date}] HTTP error {response_dw.status_code} for DW.")
            else:
                print(f"[{start_date}] No DW image found.")
        except Exception as e:
            print(f"[{start_date}] Exception processing DW: {e}")

        # 2. If DW mask is unavailable or empty, use the preloaded ESA mask
        if mask_numeric is None or np.all(mask_numeric == 0):
            print(f"[{start_date}] Using ESA static mask.")
            mask_numeric = esa_mask_numeric

        if mask_numeric is not None:
            masks[start_date] = mask_numeric
        else:
            print(f"[{start_date}] No mask could be generated.")

    return masks


