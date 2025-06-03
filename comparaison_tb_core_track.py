"""
###############################################################################
#IMPORTING PACKAGES
###############################################################################
"""

import matplotlib.pyplot as plt
plt.close('all')
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature  
import netCDF4 as ncdf
from datetime import datetime, timedelta
from functions import plot_object_mask_pcolormesh
from functions import filter_objects_by_size
import matplotlib.patheffects as path_effects




"""
###############################################################################
INITIALIZATION
###############################################################################
"""

# Define the region of interest (min and max latitudes and longitudes)
lat_min, lat_max = 41, 45
lon_min, lon_max = -2, 3  
lat_CRA, lon_CRA = 43.128, 0.366


# Step 1: Create the list of times from 14:00:00 to 17:45:00 with 15-min steps
start = datetime.strptime("13-15-00", "%H-%M-%S")
end = datetime.strptime("22-00-00", "%H-%M-%S")
time_list = []

while start <= end:
    time_str = start.strftime("%H-%M")
    time_list.append(time_str)
    start += timedelta(minutes=15)
    
    
"""
###############################################################################
CREATING COLORMAP FOR BRIGHTNESS TEMPERATURE
###############################################################################
"""

vmin, vmax = 190, 295 #Color map range
split = 230
n_colors = 256
inferno_part = int(n_colors * (split - vmin) / (vmax - vmin))
gray_part = n_colors - inferno_part

inferno = plt.get_cmap('inferno')(np.linspace(0, 1, inferno_part))
gray = plt.get_cmap('gray')(np.linspace(1, 0, gray_part))  # reversed grayscale
colors = np.vstack((inferno, gray))
custom_cmap = LinearSegmentedColormap.from_list('inferno_gray', colors)


"""
###############################################################################
IMPORTING FILES
###############################################################################
"""
grid_file = '../MSG+0000.3km.nc'
obs_file = '../SimCRA2/Concatenated/concatenated_obs_output.nc'
sim_file = '../SimCRA2/Concatenated/concatenated_sim_output.nc'

# Load lat/lon grid for obs
ncfile = ncdf.Dataset(grid_file, "r")
lat = np.ma.filled(ncfile['Latitude'][:], np.nan)
lon = np.ma.filled(ncfile['Longitude'][:], np.nan)

#Read obs
data_obs = ncdf.Dataset(obs_file, 'r')
bt_obs = np.ma.filled(data_obs.variables['IR_108'], np.nan)
objects_obs = np.ma.filled(data_obs.variables['tobjects'][:], fill_value=0)

# --- Define ROI for obs ---
mask_obs = (lat >= lat_min) & (lat <= lat_max) & \
           (lon >= lon_min) & (lon <= lon_max)
           
rows, cols = np.where(mask_obs)
i_min, i_max = rows.min(), rows.max()
j_min, j_max = cols.min(), cols.max()

# Crop obs
lat_obs = lat[i_min:i_max+1, j_min:j_max+1]
lon_obs = lon[i_min:i_max+1, j_min:j_max+1]
bt_obs = bt_obs[:, i_min:i_max+1, j_min:j_max+1]
objects_obs = objects_obs[:, i_min:i_max+1, j_min:j_max+1]

#Read sim
data_sim = ncdf.Dataset(sim_file, 'r')
bt_sim = np.ma.filled(data_sim.variables['downscaled_data'], np.nan)
objects_sim = np.ma.filled(data_sim.variables['tobjects'][:], fill_value=0)
lat_sim = np.ma.filled(data_sim.variables['latitude'][:])
lon_sim = np.ma.filled(data_sim.variables['longitude'][:])
lat_sim, lon_sim = np.meshgrid(lat_sim, lon_sim, indexing='ij')

# --- Define ROI for obs ---
mask_sim = (lat_sim >= lat_min) & (lat_sim <= lat_max) & \
           (lon_sim >= lon_min) & (lon_sim <= lon_max)
           
rows, cols = np.where(mask_sim)
i_min, i_max = rows.min(), rows.max()
j_min, j_max = cols.min(), cols.max()

# Crop sim
lat_sim = lat_sim[i_min:i_max+1, j_min:j_max+1]
lon_sim = lon_sim[i_min:i_max+1, j_min:j_max+1]
bt_sim = bt_sim[:, i_min:i_max+1, j_min:j_max+1]
objects_sim = objects_sim[:, i_min:i_max+1, j_min:j_max+1]

filtered_sim_objects = filter_objects_by_size(objects_sim, min_pixels=5)
filtered_obs_objects = filter_objects_by_size(objects_obs, min_pixels=5)


"""
###############################################################################
PLOTTING
###############################################################################
"""

n_times = bt_obs.shape[0]
for t in range(n_times):
    
    timestamp = time_list[t]
    bt_obs_t = bt_obs[t, :, :]
    filtered_objects_obs_t = filtered_obs_objects[t, : , :]
    bt_sim_t = bt_sim[t, :, :]
    filtered_objects_sim_t = filtered_sim_objects[t, : , :]
    
    fig, axs = plt.subplots(1, 2, figsize=(19, 10),
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            constrained_layout=True)

    xticks = np.arange(-2, 4, 1)
    yticks = np.arange(41, 46, 1)
    
    for ax in axs:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
 
        # Create features with desired resolution
        land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='white', facecolor='none')
        ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m',
                                             edgecolor='white', facecolor='none')
        borders = cfeature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '50m',
                                               edgecolor='white', facecolor='none')
        
        # Add features to the axis
        ax.add_feature(land, linewidth=2, zorder = 8)
        ax.add_feature(ocean, linewidth=2, zorder = 8)
        ax.add_feature(borders, linewidth=2, zorder = 8)
        
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{val}°E" for val in xticks], fontsize = 16)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{val}°N" for val in yticks], fontsize = 16)
        
        ax.plot(lon_CRA, lat_CRA,'ro',markersize=10,transform=ccrs.PlateCarree(), zorder = 15)
        ax.plot(lon_CRA, lat_CRA,'o',markersize=14,transform=ccrs.PlateCarree(), color = 'black', zorder = 14)
        txt = ax.text((lon_CRA + 0.05), (lat_CRA + 0.05), "CRA" ,color='white',fontsize=18, zorder = 15)
        txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),  # outline
                              path_effects.Normal()])
        
        
    #Common title for both plots
    fig.suptitle(f'Comparaison des températures de brillance (10.8 μm) \n avec régions et donwnscaling - {timestamp}',
                 fontsize=22, y=0.95)
    
    #Plot
    im1 = axs[0].pcolormesh(lon_obs, lat_obs, bt_obs_t, cmap=custom_cmap,
                            vmin=vmin, vmax=vmax, shading='auto', 
                            zorder = 5, transform=ccrs.PlateCarree())
    #Plot simulated
    im2 = axs[1].pcolormesh(lon_sim, lat_sim, bt_sim_t, cmap=custom_cmap,
                            vmin=vmin, vmax=vmax, shading='auto', 
                            zorder = 5, transform=ccrs.PlateCarree())
    
    #Set titles
    axs[0].set_title('$T_b$ observée (MSG)', fontsize=22)
    axs[1].set_title('$T_b$ simulée (Méso-NH)', fontsize=22)

    #Add objects
    plot_object_mask_pcolormesh(axs[0], filtered_objects_obs_t, lon_obs, lat_obs, zorder=6)
    plot_object_mask_pcolormesh(axs[1], filtered_objects_sim_t, lon_sim, lat_sim, zorder=6)
    
    # Shared colorbar
    ticks = np.arange(190, 300, 10)
    labels = np.arange(190, 300, 20)
    cbar = fig.colorbar(im2, ax=axs, orientation='vertical', shrink=0.6, pad=0.02)
    cbar.set_label('Température de brillance (K)', fontsize=18, labelpad = 20)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(i)}" if i % 20 == 0 else '' for i in ticks])
    cbar.ax.tick_params(size=3, width=0.5, labelsize=17)

    # Save figure
    plt.savefig(f"../Comparaison_track_auto/Comparison_BT_grid_down_{timestamp}.png",
                dpi=300, bbox_inches='tight')
    plt.close()