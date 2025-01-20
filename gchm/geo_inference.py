import os

geojson_path = '/path/to/geojson'
data_dir     = '/path/to/raw'
output_dir   = '/path/to/output'
sentinel_dir = os.path.join(data_dir, 'sentinel')
esa_dir      = os.path.join(data_dir, 'esa')
date_start   = '2020-01-01'
date_end     = '2020-02-01'
model_weights_path = '/path/to/model_weights.pth'


# preprocess the geojson
geometry = load_and_validate_geojson(geojson_path)

# define the spatial breakdown of the geojson into a set of tiles to be downloaded - ideally something that is 512 x 512
tiles    = geojson2tiles(geometry, date_start, date_end)

# download the tiles
download_sentinel2_tiles(tiles, sentinel_dir)
download_esa_tiles(tiles, esa_dir)

# preprocess the tiles (if needed)
preprocess_sentinel2_tiles(sentinel_dir)
preprocess_esa_tiles(esa_dir)


# create a pytorch dataset / dataloader for the tiles - make sure all preprocessing (normalization, etc.) is the same as during training
ds = create_sentinel2_dataset(sentinel_dir, input_transforms=input_transforms)
dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# load the model (copy paste from deploy.py)
architecture_collection = Architectures(args=args)
net = architecture_collection(architecture)(num_outputs=1)

net.cuda()  # move model to GPU

# load the model weights
checkpoint_path = Path(args.model_dir) / 'checkpoint.pt'
checkpoint = torch.load(checkpoint_path)
model_weights = checkpoint['model_state_dict']

# predict over the tiles
pred_dict = {}
with torch.no_grad():
    
    out_dict = {}
    for data_dict in dl:
        
        # get inputs to model (X) and metadata of the tile (i.e. location or tile_id, etc...)
        X, tile_idx = data_dict['X'], data_dict['tile_idx']
        
        y_hat, variance = net.forward(X)
        
        out_dict['predictions'].append(y_hat)
        out_dict['variance'].append(variance)
        out_dict['tile_idx'].append(tile_idx)
        
        # perform any postprocessing here ...
        

# recompose the predictions and variances into geospatial rasters
y_hat_raster    = recompose_tiles(out_dict['predictions'], out_dict['tile_idx'], tiles, geometry)
variance_raster = recompose_tiles(out_dict['variance'], out_dict['tile_idx'], tiles, geometry)

# save the rasters
save_array_as_geotif(out_path=os.path.join(output_dir, 'y_hat.tif'), array=y_hat_raster, tile_info=tiles)
save_array_as_geotif(out_path=os.path.join(output_dir, 'variance.tif'), array=variance_raster, tile_info=tiles)


# visualize the rasters
plot_rasters(y_hat_raster, variance_raster)


