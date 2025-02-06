from datetime import datetime, timedelta
import glob
import os
import numpy as np
import json
import torch
from pathlib import Path
import sys
import copy

sys.path.append('/content/drive/MyDrive/GCHM_geo_inferences/')

from gchm.models.architectures import Architectures
from gchm.utils.transforms import Normalize
from gchm.datasets.dataset_sentinel2_deploy import Sentinel2Deploy
from gchm.utils.gdal_process import save_array_as_geotif
from gchm.utils.parser import load_args_from_json, str2bool, str_or_none
from ee_preprocess import initialize_gee, load_and_validate_geojson,get_square_encompassing_polygon,create_composite, export_image_to_drive,get_dlc_mask
from deploy import predict,setup_parser


## Setup the script parameters
# Parse deploy arguments
parser = setup_parser()
args, unknown = parser.parse_known_args()

# Initialize GEE
initialize_gee(args.service_account_key)


## AoI Inputs, and data retrieval
# Load and validate the GeoJSON
aoi = load_and_validate_geojson(args.geojson_path)

# Get a square that encompass the AoI
geometry = get_square_encompassing_polygon(aoi)

# Get DLC masks for all the dates in the array (dict[date,mask])
dlc_masks = get_dlc_mask(geometry, args.date_array)


# Export images for each date to Google Drive (if needed)
for date in args.date_array:
    lowest_cloud_image = create_composite(date, geometry)
    reprojected = lowest_cloud_image.reproject(crs='EPSG:4326', scale=10)
    #export_image_to_drive(reprojected, f"{args.aoi_name}_{date}", args.tif_dir, geometry)
    print(f"Pre-process completed for {date}")

##  Model parameter
# Take randomly one of the 5 CNNs
num_model = np.random.choice(5)
print('Number of the model:', num_model)

# Set the path to the model weight
model_dir_path = os.path.join(args.model_dir, f"model_{num_model}", args.finetune_strategy)

# Set model-related arguments
args.model_id = num_model
args.model_dir = model_dir_path
if args.input_lat_lon:
    args.channels = 15 

args_dict = vars(args)
args_dict_deploy = copy.deepcopy(args_dict)

# Load args from experiment dir with full train set
print("Loading args from trained models directory...")
args_saved = load_args_from_json(os.path.join(args.model_dir, 'args.json'))
# Update args with model_dir args
args_dict.update(args_saved)
# Update args with deploy args
args_dict.update(args_dict_deploy)


# Setup the device
DEVICE = torch.device("cuda:0")
print('DEVICE: ', DEVICE, torch.cuda.get_device_name(0))


# Load training statistics
train_input_mean = np.load(os.path.join(model_dir_path, 'train_input_mean.npy'))
train_input_std = np.load(os.path.join(model_dir_path, 'train_input_std.npy'))
train_target_mean = np.load(os.path.join(model_dir_path, 'train_target_mean.npy'))
train_target_std = np.load(os.path.join(model_dir_path, 'train_target_std.npy'))

print("Training statistics loaded successfully:")
print(f"train_input_mean: {train_input_mean}")
print(f"train_input_std: {train_input_std}")
print(f"train_target_mean: {train_target_mean}")
print(f"train_target_std: {train_target_std}")


# Setup input transforms
input_transforms = Normalize(mean=train_input_mean, std=train_input_std)

# Path to the images to deploy
image_paths = glob.glob(f"{args.image_path}/*.tif")
print(image_paths)

print(f"Path to images for prediction : {args.image_path}") 
print("Content of the folder:", os.listdir(args.image_path)) 



# Load model architecture
architecture_collection = Architectures(args=args)
net = architecture_collection(args.architecture)(num_outputs=1)

net.cuda()  # Move model to GPU

# Load latest weights from checkpoint file
print('Loading model weights from latest checkpoint ...')
checkpoint_path = Path(model_dir_path) / 'checkpoint.pt'
checkpoint = torch.load(checkpoint_path)
model_weights = checkpoint['model_state_dict']

for image_path in image_paths:
    print(f"Processing image: {image_path}")
    
    # Extract date from image name (assuming the image name contains the date in YYYY-MM-DD format)
    base_filename = os.path.basename(image_path)
    base_filename_without_extension = os.path.splitext(base_filename)[0]  # retire l'extension .tif

    date_str = base_filename_without_extension.split('_')[1]  # Adjust according to your naming convention
    print(f"Image date extracted: {date_str}")
    
    # Load dataset for each image
    ds_pred = Sentinel2Deploy(
        path=image_path,
        input_transforms=input_transforms,
        input_lat_lon=args.input_lat_lon,
        patch_size=args.batch_size,
        border=8
    )


    # Predict for this image
    pred_dict = predict(
        model=net,
        args=args,
        model_weights=model_weights,
        ds_pred=ds_pred,
        batch_size=args.batch_size,
        num_workers=4,
        train_target_mean=train_target_mean,
        train_target_std=train_target_std
    )

    # Recompose predictions and apply DLC mask for this image
    for k in pred_dict:
        recomposed_tiles = ds_pred.recompose_patches(pred_dict[k], out_type=np.float32)
        if date_str in dlc_masks:
            dlc_mask = dlc_masks[date_str]
            print(f"Unique values in DLC mask for {date_str}: {np.unique(dlc_mask)}")
        
            print("Shape of recomposed_tiles: ", recomposed_tiles.shape)
            print("Shape of dlc_mask: ", dlc_mask.shape)
            dlc_mask = np.array(dlc_mask, dtype=np.float32)  # Convert to a regular NumPy array

            # Appliquer le masque (mettre à NaN ou 0 les valeurs masquées)
            recomposed_tiles[dlc_mask == 0] = np.NaN  # Ou une autre valeur selon ton besoin

            # Sauvegarder l'image masquée
            tif_path = os.path.join(args.deploy_dir, f"{base_filename_without_extension}_{k}.tif")
            print(f"Saving to: {tif_path}")
            save_array_as_geotif(tif_path, recomposed_tiles, ds_pred.tile_info)

        else :
            tif_path = os.path.join(args.deploy_dir, f"{base_filename_without_extension}_{k}.tif")
            print(f"Saving with no matching mask: {tif_path}")
            save_array_as_geotif(tif_path, recomposed_tiles, ds_pred.tile_info)

"""    
    # Save lat/lon masks if needed
    if args.save_latlon_masks:
        lat_path = os.path.join(args.deploy_dir, f"{base_filename_without_extension}_lat.tif")
        lon_path = os.path.join(args.deploy_dir, f"{base_filename_without_extension}_lon.tif")

        save_array_as_geotif(lat_path, ds_pred.lat_mask[ds_pred.border:-ds_pred.border, ds_pred.border:-ds_pred.border], ds_pred.tile_info)
        save_array_as_geotif(lon_path, ds_pred.lon_mask[ds_pred.border:-ds_pred.border, ds_pred.border:-ds_pred.border], ds_pred.tile_info)

    # Optionally remove original image after processing
    if args.remove_image_after_pred:
        os.remove(image_path)
        print(f"Removed original image: {image_path}")
"""