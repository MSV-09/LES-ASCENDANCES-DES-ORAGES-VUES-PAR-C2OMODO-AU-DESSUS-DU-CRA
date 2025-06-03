import numpy as np
import xarray as xr
from pyresample import geometry, kd_tree
import netCDF4 as ncdf

def plot_data(data, data_obs, title):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    vmin = 190
    vmax = 310 

    threshold = 240

    colors_under_threshold = plt.cm.inferno(np.linspace(0, 1, threshold - vmin))
    colors_above_threshold = np.flipud(plt.cm.gray(np.linspace(0, 1, vmax - threshold)))
    colors = np.vstack((colors_under_threshold, colors_above_threshold))
    new_cmap = mcolors.ListedColormap(colors, name='inferno_gray')
    bounds = np.linspace(vmin, vmax, vmax - vmin + 1)
    norm = mcolors.BoundaryNorm(bounds, new_cmap.N)

    plt.figure(figsize=(12, 12))
    plt.subplot(1, 2, 1)
    plt.title("Downscaled Data")
    plt.imshow(data, cmap=new_cmap, norm=norm)
    plt.subplot(1, 2, 2)
    plt.title("Original Data")
    plt.imshow(data_obs, cmap=new_cmap, norm=norm)
    cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=new_cmap), cax=cbar_ax)
    plt.show()
    

def downscale(sim_file, obs_file, grid_file):
    def crop_data(data, lat_indices, lon_indices):
        return data[lat_indices.min():lat_indices.max() + 1,
                    lon_indices.min():lon_indices.max() + 1]

    dataset = xr.open_dataset(sim_file)
    data_sim = dataset['MET7_IRBT'].values[0]
    lat_sim, lon_sim = dataset['latitude'].values, dataset['longitude'].values
    lat_min_sim, lat_max_sim = lat_sim.min(), lat_sim.max()
    lon_min_sim, lon_max_sim = lon_sim.min(), lon_sim.max()

    grid_dataset = ncdf.Dataset(grid_file, "r")
    obs_data = ncdf.Dataset(obs_file, "r")
    lat_obs, lon_obs = grid_dataset['Latitude'][:][:, :], grid_dataset['Longitude'][:][:, :]
    data_obs = obs_data['IR_108'][:][:, :]
    
    lat_mask = (lat_obs >= lat_min_sim) & (lat_obs <= lat_max_sim)
    lon_mask = (lon_obs >= lon_min_sim) & (lon_obs <= lon_max_sim)
    lat_indices = np.where(lat_mask.any(axis=1))[0]
    lon_indices = np.where(lon_mask.any(axis=0))[0]

    # Crop the observation data to the simulation grid
    lat_obs_cropped = crop_data(lat_obs, lat_indices, lon_indices)
    lon_obs_cropped = crop_data(lon_obs, lat_indices, lon_indices)
    data_obs_cropped = crop_data(data_obs, lat_indices, lon_indices)

    # Create a SwathDefinition object for the observation data
    area_sim = geometry.SwathDefinition(lons=lon_sim, lats=lat_sim)
    area_obs = geometry.SwathDefinition(lons=lon_obs_cropped, lats=lat_obs_cropped)

    # Resampling using nearest neighbor interpolation between the simulation and observation grids
    data_on_obs = kd_tree.resample_nearest(area_sim, data_sim, area_obs,
                                           radius_of_influence=3000, fill_value=np.nan)

    valid_mask = ~np.isnan(data_on_obs)
    valid_lat_indices = np.where(valid_mask.any(axis=1))[0]
    valid_lon_indices = np.where(valid_mask.any(axis=0))[0]

    # Crop the data to the valid indices : Center the data on the original simulation grid
    data_on_obs_cropped = crop_data(data_on_obs, valid_lat_indices, valid_lon_indices)
    lat_obs_cropped_final = crop_data(lat_obs_cropped, valid_lat_indices, valid_lon_indices)
    lon_obs_cropped_final = crop_data(lon_obs_cropped, valid_lat_indices, valid_lon_indices)
    data_obs_cropped_final = crop_data(data_obs_cropped, valid_lat_indices, valid_lon_indices)

    plot_data(data_on_obs_cropped, data_obs_cropped_final, "Downscaled Data")

    return valid_lat_indices, valid_lon_indices



if __name__ == "__main__":
    import sys
    import time
    
    if len(sys.argv) < 3:
        print("Usage : python downscale.py <file.nc> <obs_file.hdf> <grid_file.nc>")
        sys.exit(1)

    input_file = sys.argv[1]
    obs_file = sys.argv[2]
    grid_file = sys.argv[3]

    print("Executing downscale with numpy...")
    start_time = time.time()
    down_file, down_coords = downscale(input_file, obs_file, grid_file)
    time_down = time.time() - start_time
    print(f"Execution time with numpy : {time_down:.4f} seconds")

    # if '-s' in sys.argv:
    print("Saving downscaled data to NetCDF...")

    downscaled_dataset = xr.Dataset(
        {
            "downscaled_data": (["latitude", "longitude"], down_file),
        },
        coords={
            "latitude": (["latitude"], down_coords[0][:, 0]),
            "longitude": (["longitude"], down_coords[1][0, :]),
        },
    )
    downscaled_dataset.attrs['description'] = 'Downscaled data from Meso-NH simulation to satellite MSG grid'

    downscaled_dataset.to_netcdf("downscaled_output.nc")
    print("Downscaled data saved to downscaled_output.nc")