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
import os
import re
from datetime import datetime, timedelta
import matplotlib.patheffects as path_effects


"""
###############################################################################
#SOME DEFINITIONS
###############################################################################
"""

# Define the region of interest (min and max latitudes and longitudes)
lat_min, lat_max = 41, 45
lon_min, lon_max = -2, 3  

lat_CRA, lon_CRA = 43.128, 0.366


"""
###############################################################################
#CREATING COLORMAP
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
#IMPORTING FILES
###############################################################################

###############################################################################
#IMPORT GRID FILE
###############################################################################
"""

grid_file_path = '../MSG+0000.3km.nc'

"""
###############################################################################
#IMPORT SIM FILES
###############################################################################
"""
#filename_0 are for extracting data that are the same for every file
sim_path = '../SimCRA2/OUT/'
sim_filename0 = sim_path + 'FLY15.1.SEP13.OUT.001.nc'

# List all .nc files in the directory
sim_files = [f for f in os.listdir(sim_path) if f.endswith('.nc')]

# Regex pattern to extract the number (e.g., 083 from FLY15.1.SEP13.OUT.083.nc)
sim_pattern = re.compile(r"FLY15\.1\.SEP13\.OUT\.(\d+)\.nc$")

# Filter files with:
# - a valid number
# - odd number
# - number between a and b
filtered_files = []
for f in sim_files:
    match = sim_pattern.match(f)
    if match:
        num = int(match.group(1)) #File number
        if 1 <= num <= 93 and num % 2 != 0:
            filtered_files.append(f)

# Sort by extracted number with a function lambda
filtered_files.sort(key=lambda f: int(sim_pattern.match(f).group(1)))

# Add full path to each file
sim_files = [os.path.join(sim_path, f) for f in filtered_files]

"""
###############################################################################
#IMPORT OBS FILES
###############################################################################
"""

obs_path = '../20230529/OBS2'
obs_filename0 = '../20230529/OBS2/Mmultic3kmNC4_msg03_001.nc'

# List all .nc files in the directory
obs_files = [f for f in os.listdir(obs_path) if f.endswith('.nc')]

# Regex pattern to extract the number (e.g., 001 from Mmultic3kmNC4_msg03_001.nc)
obs_pattern = re.compile(r"Mmultic3kmNC4_msg03_(\d+)\.nc$")

found_numbers = []
filtered_files = []  # Ensure filtered_files is initialized
for f in os.listdir(obs_path):
    if f.endswith('.nc'):
        match = obs_pattern.match(f)
        if match:
            num = int(match.group(1))
            if 54 <= num <= 89:
                found_numbers.append(num)
                filtered_files.append(f)

# Sort by extracted number with a function lambda
filtered_files.sort(key=lambda f: int(obs_pattern.match(f).group(1)) if obs_pattern.match(f) else float('inf'))

# Add full path to each file
obs_files = [os.path.join(obs_path, f) for f in filtered_files]

start = datetime.strptime("13-15-00", "%H-%M-%S")
end = datetime.strptime("22-00-00", "%H-%M-%S")
time_list = []

while start <= end:
    time_str = start.strftime("%H-%M")
    time_list.append(time_str)
    start += timedelta(minutes=15)
    
"""
###############################################################################
#READING FILE
###############################################################################
"""
for obs_file, sim_file, timestamp in zip(obs_files, sim_files, time_list):

    # === Load observation data ===
    data_obs = ncdf.Dataset(obs_file, 'r')
    brightness_temp = data_obs.variables['IR_108']
    bt_obs = brightness_temp[:] 
    bt_obs = np.ma.filled(bt_obs, np.nan)

    ncfile = ncdf.Dataset(grid_file_path, "r")
    lat_obs = np.ma.filled(ncfile['Latitude'][:], np.nan) # Replace missing values with NaN
    lon_obs = np.ma.filled(ncfile['Longitude'][:], np.nan)
    
    # === Define ROI ===
    mask = (lat_obs >= lat_min) & (lat_obs <= lat_max) & (lon_obs >= lon_min) \
    & (lon_obs <= lon_max)
    
    indices = np.where(mask)
    i_min, i_max = indices[0].min(), indices[0].max()
    j_min, j_max = indices[1].min(), indices[1].max()
    lat_obs = lat_obs[i_min:i_max+1, j_min:j_max+1]
    lon_obs = lon_obs[i_min:i_max+1, j_min:j_max+1]
    bt_obs = bt_obs[i_min:i_max+1, j_min:j_max+1]

    # === Load simulation data ===
    data_sim = ncdf.Dataset(sim_file, "r")
    bt_sim = data_sim.variables['MET7_IRBT'][0, :]
    lon_sim = data_sim.variables['longitude'][:]
    lat_sim = data_sim.variables['latitude'][:]
        

###############################################################################
#PLOTTING FILE
###############################################################################


    fig, axs = plt.subplots(1, 2, figsize=(19, 10),
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            constrained_layout=True)

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
        
        ax.plot(lon_CRA, lat_CRA,'ro',markersize=10,transform=ccrs.PlateCarree(), zorder = 15)
        ax.plot(lon_CRA, lat_CRA,'o',markersize=14,transform=ccrs.PlateCarree(), color = 'black', zorder = 14)
        txt = ax.text((lon_CRA + 0.05), (lat_CRA + 0.05), "CRA" ,color='white',fontsize=18, zorder = 15)
        txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),  # outline
                              path_effects.Normal()])
        
        
    #Common title for both plots
    fig.suptitle(f'Comparaison des températures de brillance (10.8 μm) - {timestamp}',
                 fontsize=22, y=0.95)
    
    #Plot
    im1 = axs[0].pcolormesh(lon_obs, lat_obs, bt_obs, cmap=custom_cmap,
                            vmin=vmin, vmax=vmax, shading='auto',
                            zorder = 5, transform=ccrs.PlateCarree())
    
    im2 = axs[1].pcolormesh(lon_sim, lat_sim, bt_sim, cmap=custom_cmap,
                            vmin=vmin, vmax=vmax, shading='auto', 
                            zorder = 5, transform=ccrs.PlateCarree())
    
    #Contours
    contours = axs[0].contour(lon_obs, lat_obs, bt_obs, levels=[220], zorder = 6,
                              colors='k', linewidths=1, transform=ccrs.PlateCarree())
    axs[0].clabel(contours, inline=True, fontsize=8, fmt="%.0f", zorder = 7)
    
    
    contours = axs[1].contour(lon_sim, lat_sim, bt_sim, levels=[220], zorder = 6,
                              colors='k', linewidths=1, transform=ccrs.PlateCarree())
    axs[1].clabel(contours, inline=True, fontsize=8, fmt="%.0f", zorder = 7)
    
    
    #Title
    axs[0].set_title('$T_b$ observée (MSG)', fontsize=20)
    axs[1].set_title('$T_b$ simulée (Méso-NH)', fontsize=20)
            

    #Ticks
    axs[0].set_xticks(np.arange(-2, 4, 1))
    axs[0].set_xticklabels([f"{val}°E" for val in np.arange(-2, 4, 1)], fontsize = 14)
    axs[0].set_yticks(np.arange(41, 46, 1))
    axs[0].set_yticklabels([f"{val}°N" for val in np.arange(41, 46, 1)], fontsize = 14)
    
    axs[1].set_xticks(np.arange(-2, 4, 1))
    axs[1].set_xticklabels([f"{val}°E" for val in np.arange(-2, 4, 1)], fontsize = 14)
    axs[1].set_yticks(np.arange(41, 46, 1))
    axs[1].set_yticklabels([f"{val}°N" for val in np.arange(41, 46, 1)], fontsize = 14)

    
    # Shared colorbar
    ticks = np.arange(190, 300, 10)
    labels = np.arange(190, 300, 20)
    cbar = fig.colorbar(im2, ax=axs, orientation='vertical', shrink=0.6, pad=0.02)
    cbar.set_label('Température de brillance (K)', fontsize=18, labelpad = 20)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(i)}" if i % 20 == 0 else '' for i in ticks])
    cbar.ax.tick_params(size=3, width=0.5, labelsize=16)

    # Save figure
    plt.savefig(f"../Comparaison/Comparison_BT_grid_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()



