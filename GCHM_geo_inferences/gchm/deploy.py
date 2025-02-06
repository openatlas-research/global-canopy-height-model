from osgeo import gdal, osr, ogr, gdalconst
import os
import sys
import numpy as np
import argparse
from pathlib import Path
import copy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


from gchm.models.architectures import Architectures
from gchm.utils.transforms import Normalize, NormalizeVariance, denormalize
from gchm.datasets.dataset_sentinel2_deploy import Sentinel2Deploy
from gchm.utils.gdal_process import save_array_as_geotif
from gchm.utils.parser import load_args_from_json, str2bool, str_or_none


import argparse
import os
import copy

import argparse

def setup_parser():
    parser = argparse.ArgumentParser()

    # Authentication and paths
    parser.add_argument("--service_account_key", required=True, 
                        help="Service account key for Google Earth Engine")
    parser.add_argument("--geojson_path", required=True, 
                        help="Path to the GeoJSON file defining the AOI")
    parser.add_argument("--tif_dir", required=True, 
                        help="Output directory for TIFF files")

    # Dates and area of interest
    parser.add_argument("--date_array",nargs='+', type=str, required=True, 
                        help="List of acquisition dates (format YYYY-MM-DD)")
    parser.add_argument("--aoi_name", required=True, 
                        help="Name of the area of interest")

    # Model and deployment parameters
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of the square patches used for prediction")
    parser.add_argument("--model_dir", required=True, 
                        help="Directory of the pre-trained model")
    parser.add_argument("--image_path", required=True, 
                        help="Path to the folder containing Sentinel-2 images for deployment")
    parser.add_argument("--deploy_dir", required=True, 
                        help="Output directory to save predictions")
    parser.add_argument("--num_models", type=int, default=5, 
                        help="Number of models in the ensemble")
    parser.add_argument("--finetune_strategy", choices=['', None,
                        'FT_ALL_CB', 'FT_L_CB', 'RT_L_CB',
                        'FT_ALL_SRCB', 'FT_L_SRCB', 'RT_L_SRCB', 
                        'FT_Lm_SRCB', 'RT_Lm_SRCB', 'RT_L_IB',
                        'ST_geoshift_IB', 'ST_geoshiftscale_IB'], 
                        default='FT_Lm_SRCB', help="Fine-tuning strategy for the model")

    # Additional options
    parser.add_argument("--save_latlon_masks", default=False, 
                        help="Save lat/lon masks used for prediction")
    parser.add_argument("--remove_image_after_pred", default=False, 
                        help="Delete the image after saving the prediction")

    # Additional model parameters
    parser.add_argument("--channels", type=int, default=12, 
                        help="Number of input channels")
    parser.add_argument("--return_variance", default=True, 
                        help="Return the model variance")
    parser.add_argument("--input_lat_lon", default=True, 
                        help="Add lat/lon coordinates as input to the model")
    parser.add_argument("--manual_init", default=False, 
                        help="Enable manual weight initialization")
    parser.add_argument("--freeze_last_mean", default=False, 
                        help="Freeze the last mean layer")
    parser.add_argument("--freeze_last_var", default=False, 
                        help="Freeze the last variance layer")
    parser.add_argument("--geo_shift", default=False, 
                        help="Enable geographic shift")
    parser.add_argument("--geo_scale", default=False, 
                        help="Enable geographic scaling")
    parser.add_argument("--separate_lat_lon", default=False, 
                        help="Separate lat/lon channels")

    return parser




def predict(model, args, model_weights=None,
            ds_pred=None, batch_size=1, num_workers=8,
            train_target_mean=0, train_target_std=1):
    DEVICE = torch.device("cuda:0")

    train_target_mean = torch.tensor(train_target_mean)
    train_target_std = torch.tensor(train_target_std)

    dl_pred = DataLoader(ds_pred, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model.load_state_dict(model_weights)
    model.eval()

    pred_dict = {'predictions': []}
    if args.return_variance:
        pred_dict['std'] = []

    with torch.no_grad():
        for step, data_dict in enumerate(tqdm(dl_pred, ncols=100, desc='pred', file=sys.stdout)):
            inputs = data_dict[args.input_key]
            inputs = inputs.to(DEVICE, non_blocking=True)

            if args.return_variance:
                predictions, variances = model.forward(inputs)
                std = torch.sqrt(variances)
                pred_dict['std'].extend(list(std.cpu()))
            else:
                predictions = model.forward(inputs)
            pred_dict['predictions'].extend(list(predictions.cpu()))

        for key in pred_dict.keys():
            if pred_dict[key]:
                pred_dict[key] = torch.stack(pred_dict[key], dim=0)
                print("val_dict['{}'].shape: ".format(key), pred_dict[key].shape)

    if args.normalize_targets:
        pred_dict['predictions'] = denormalize(pred_dict['predictions'], train_target_mean, train_target_std)
        if args.return_variance:
            pred_dict['std'] *= train_target_std

    for key in pred_dict.keys():
        pred_dict[key] = pred_dict[key].data.cpu().numpy()

    return pred_dict
