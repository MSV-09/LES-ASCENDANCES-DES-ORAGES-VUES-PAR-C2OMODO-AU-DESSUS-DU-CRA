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
from netCDF4 import Dataset
import os
import re
from datetime import datetime, timedelta
import matplotlib.colors as mcolors
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




#Create the list of times from 13:15:00 to 21:45:00 with 15-min steps
start = datetime.strptime("13-15-00", "%H-%M-%S")
end = datetime.strptime("21-45-00", "%H-%M-%S")
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

###############################################################################
IMPORT GRID FILE
###############################################################################
"""


grid_file_path = '../MSG+0000.3km.nc'


"""
###############################################################################
IMPORT SIM FILES
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
        if 2 <= num <= 71:
            filtered_files.append(f)

# Sort by extracted number with a function lambda
filtered_files.sort(key=lambda f: int(sim_pattern.match(f).group(1)))

# Add full path to each file
files = [os.path.join(sim_path, f) for f in filtered_files]


"""
###############################################################################
EXTRACT DATA
###############################################################################
"""


# ---- Step 1: Extract Tb, lat, lon ----
bt_list = []
lat_list = []
lon_list = []
WT_list = []

downscale_factor = 3  # You can change this globally

for i in range(0, len(files) - 1, 2):
    with Dataset(files[i], 'r') as nc1, Dataset(files[i+1], 'r') as nc2:
        # Read variables from first file
        bt1 = nc1.variables['aos_3250BT'][0, :]
        lat = nc1.variables['latitude'][:]
        lon = nc1.variables['longitude'][:]
        WT1 = nc1.variables['WT'][0, :, :, :]

        # Read WT from second file
        WT2 = nc2.variables['WT'][0, :, :, :]
        bt2 = nc2.variables['aos_3250BT'][0, :]

        # --- Define ROI using lat/lon ---
        mask_sim = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
        rows, cols = np.where(mask_sim)
        if len(rows) == 0 or len(cols) == 0:
            print("No sim data in ROI, skipping.")
            continue
        i_min, i_max = rows.min(), rows.max()
        j_min, j_max = cols.min(), cols.max()

        # Crop spatial dimensions
        lat = lat[i_min:i_max+1, j_min:j_max+1]
        lon = lon[i_min:i_max+1, j_min:j_max+1]
        bt1 = bt1[i_min:i_max+1, j_min:j_max+1]
        WT1 = WT1[:, i_min:i_max+1, j_min:j_max+1]
        WT2 = WT2[:, i_min:i_max+1, j_min:j_max+1]
        bt2 = bt2[i_min:i_max+1, j_min:j_max+1]
        
        
        

        # === Compute WT_max averaged over the pair ===
        WT1_max = WT1.max(axis=0)
        WT2_max = WT2.max(axis=0)
        WT_avg_max = (WT1_max + WT2_max) / 2
        
        delta_tb = (bt2 - bt1) / (2*60) #Convert in K/s

    bt_list.append(delta_tb.data)
    lat_list.append(lat.data)
    lon_list.append(lon.data)
    WT_list.append(WT_avg_max)

    print(f"Pair: {files[i]} & {files[i+1]} → WT_avg_max range = {WT_avg_max.min():.2f} to {WT_avg_max.max():.2f}")

print(len(bt_list))

if len(files) % 2 != 0:
    print("Warning: Odd number of files, last one will be skipped.")


for idx in range(len(WT_list)):
    delta_tb = bt_list[idx]
    lat = lat_list[idx]
    lon = lon_list[idx]
    timestamp = time_list[idx]
    WT = WT_list[idx]
    
    
    # Original colormap
    base_cmap = plt.get_cmap('seismic', 256)
    cmap_array = base_cmap(np.linspace(0, 1, 256))
    
    # Define value range
    vmin, vmax = -0.2, 0.2
    threshold = 0.01
    
    # Find colormap indices for the threshold
    def value_to_index(val, vmin, vmax, n_colors):
        return int((val - vmin) / (vmax - vmin) * (n_colors - 1))
    
    low_idx = value_to_index(-threshold, vmin, vmax, 256)
    high_idx = value_to_index(threshold, vmin, vmax, 256)
    
    # Replace values around zero with white
    cmap_array[low_idx:high_idx+1] = [1, 1, 1, 1]  # RGBA white
    
    # Create new ListedColormap
    bt_cmap = mcolors.ListedColormap(cmap_array)


    # Create a figure with two subplots: left for ΔT_b and right for WT_max
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'projection': ccrs.PlateCarree()})  # Define projection for both subplots
    ax1, ax2 = axes[0], axes[1]

    # Plot ΔT_b (temperature difference) on the left subplot
    mesh1 = ax1.pcolormesh(lon, lat, delta_tb, cmap=bt_cmap, vmin=-0.2, vmax=0.2, shading='auto', transform=ccrs.PlateCarree())

    
    # Original colormap
    base_cmap = plt.get_cmap('OrRd', 256)
    cmap_array = base_cmap(np.linspace(0, 1, 256))
    
    # Define value range
    vmin, vmax = 0,  20
    threshold = 2
    
    # Find colormap indices for the threshold
    def value_to_index(val, vmin, vmax, n_colors):
        return int((val - vmin) / (vmax - vmin) * (n_colors - 1))
    
    low_idx = 0
    high_idx = value_to_index(threshold, vmin, vmax, 256)
    
    # Replace values around zero with white
    cmap_array[low_idx:high_idx+1] = [1, 1, 1, 1]  # RGBA white
    
    # Create new ListedColormap
    WT_cmap = mcolors.ListedColormap(cmap_array)
    
    # Plot WT_max on the right subplot
    mesh2 = ax2.pcolormesh(lon, lat, WT_list[idx], cmap=WT_cmap, vmin = vmin,
                           vmax = vmax, shading='auto')

    # Add coastlines and context features for both subplots
    for ax in [ax1, ax2]:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())  # Apply to axes with PlateCarree projection
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        
        ax.plot(lon_CRA, lat_CRA,'o',markersize=14,transform=ccrs.PlateCarree(), color = 'black', zorder = 14)
        ax.plot(lon_CRA, lat_CRA,'ro',markersize=10,transform=ccrs.PlateCarree(), zorder = 15)
        
        txt = ax.text((lon_CRA + 0.05), (lat_CRA + 0.05), "CRA" ,color='white',fontsize=18, zorder = 15)
        txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),  # outline
                              path_effects.Normal()])

    # Colorbars
    
    cbar1 = plt.colorbar(mesh1, ax=ax1, orientation='vertical', shrink=0.8, pad=0.05)
    cbar1.ax.tick_params(labelsize=12)
    cbar1.set_label(r"$\frac{\Delta T_b}{\Delta t}$ à 325 GHz (K s$^{-1}$)", fontsize=18)

    cbar2 = plt.colorbar(mesh2, ax=ax2, orientation='vertical', shrink=0.8, pad=0.05)
    cbar2.ax.tick_params(labelsize=12)
    cbar2.set_label("w$_{max}^{col}$  (m s$^{-1})$", fontsize=18)
    



    # Titles and labels
    fig.suptitle(   fr"$\frac{{\Delta T_b}}{{\Delta t}}$ et $w_{{max}}$ à {timestamp} "
    r"$(\Delta t = 1\,\mathrm{min})$",
                 fontsize=18, y=0.9)

    # Common labels for both subplots
    ax1.set_xlabel("Longitude", fontsize=16)
    ax1.set_ylabel("Latitude", fontsize=16)
    
    ax2.set_xlabel("Longitude", fontsize=16)
    ax2.set_ylabel("Latitude", fontsize=16)

    
    ax1.grid(True, alpha = 0.7)
    ax2.grid(True, alpha = 0.7)

    # Set ticks and tick labels
    for ax in [ax1, ax2]:
        ax.set_xticks(np.arange(-2, 4, 1))
        ax.set_xticklabels([f"{val}°E" for val in np.arange(-2, 4, 1)])
        ax.set_yticks(np.arange(41, 46, 1))
        ax.set_yticklabels([f"{val}°N" for val in np.arange(41, 46, 1)])
        
        # Set tick label font size manually
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(14)

    # Adjust layout for better spacing between subplots
    plt.tight_layout()

    plt.savefig(f"../delta_Tb_map/Delta_Tb_WT_max_map_{timestamp}.png", dpi = 300)
    # Show the plot
    plt.show()
