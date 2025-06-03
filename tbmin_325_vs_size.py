"""
###############################################################################
#IMPORTING PACKAGES
###############################################################################
"""

import matplotlib.pyplot as plt
plt.close('all')
from matplotlib.colors import LinearSegmentedColormap
import numpy as np 
import netCDF4 as ncdf
from datetime import datetime, timedelta
from functions import filter_objects_by_size, count_objects, plot_tbmin_vs_size_by_status
from functions import plot_tbmin325_vs_size_by_status
import matplotlib.dates as mdates


"""
###############################################################################
INITIALIZATION
###############################################################################
"""

# Define the region of interest (min and max latitudes and longitudes)
lat_min, lat_max = 41, 45
lon_min, lon_max = -2, 3  



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
sim_file = '../SimCRA2/Concatenated/concatenated_sim_objects.nc'

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
#bt_sim = np.ma.filled(data_sim.variables['downscaled_data'], np.nan)
objects_sim = np.ma.filled(data_sim.variables['tobjects'][:], fill_value=0)
lat_sim = np.ma.filled(data_sim.variables['latitude'][:])
lon_sim = np.ma.filled(data_sim.variables['longitude'][:])
bt_325 = np.ma.filled(data_sim.variables['aos_3250BT'][:], np.nan)

# --- Define ROI for obs ---
mask_sim = (lat_sim >= lat_min) & (lat_sim <= lat_max) & \
           (lon_sim >= lon_min) & (lon_sim <= lon_max)
           
rows, cols = np.where(mask_sim)
i_min, i_max = rows.min(), rows.max()
j_min, j_max = cols.min(), cols.max()

# Crop sim
lat_sim = lat_sim[i_min:i_max+1, j_min:j_max+1]
lon_sim = lon_sim[i_min:i_max+1, j_min:j_max+1]
objects_sim = objects_sim[:, i_min:i_max+1, j_min:j_max+1]
bt_325 = bt_325[:, i_min:i_max+1, j_min:j_max+1]

filtered_sim_objects = filter_objects_by_size(objects_sim, min_pixels=5)
filtered_obs_objects = filter_objects_by_size(objects_obs, min_pixels=5)

# === Print object sizes after cropping ===
from collections import Counter

def get_object_sizes(label_array):
    """
    Returns a dictionary {object_id: size (in pixels)} for a labeled array.
    """
    labels = label_array[label_array != 0]  # exclude background
    return dict(Counter(labels))

# --- Observation object sizes ---
sizes_obs = get_object_sizes(filtered_obs_objects)
if sizes_obs:
    print(f"\n[OBS] Number of objects: {len(sizes_obs)}")
    print(f"[OBS] Min size: {min(sizes_obs.values())} px, Max size: {max(sizes_obs.values())} px")
else:
    print("\n[OBS] No objects found after cropping.")

# --- Simulation object sizes ---
sizes_sim = get_object_sizes(filtered_sim_objects)
if sizes_sim:
    print(f"\n[SIM] Number of objects: {len(sizes_sim)}")
    print(f"[SIM] Min size: {min(sizes_sim.values())} px, Max size: {max(sizes_sim.values())} px")
else:
    print("\n[SIM] No objects found after cropping.")


def print_min_object_size(label_3d, name=""):
    import numpy as np
    all_sizes = []
    for t in range(label_3d.shape[0]):
        labels = label_3d[t]
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            if label != 0:
                all_sizes.append(count)
    if all_sizes:
        print(f"{name} - Min object size (px): {min(all_sizes)}")
    else:
        print(f"{name} - No objects found.")




"""
###############################################################################
COUNTING
###############################################################################
"""


# Initialize an empty list to store the number of objects for each timestep
simulation_object_counts = []
observation_object_counts = []

# Assuming `simulation_masks` and `observation_masks` are lists of 2D arrays for each timestep
for sim_mask, obs_mask in zip(filtered_sim_objects, filtered_obs_objects):
    # Count objects in both simulation and observation masks
    sim_num_objects = count_objects(sim_mask)
    obs_num_objects = count_objects(obs_mask)
    
    # Store the counts in the respective lists
    simulation_object_counts.append(sim_num_objects)
    observation_object_counts.append(obs_num_objects)

# === Convert string timestamps to datetime ===
time_format = "%H-%M"  # Adjust this if your format is different
time_dt = [datetime.strptime(t, time_format) for t in time_list]

# === Filter only full hours ===
full_hours = [t for t in time_dt if t.minute == 0]

# === Plot ===
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(time_dt, simulation_object_counts, label='Simulated', marker='o', color='red',
        markersize = 10)

ax.plot(time_dt, observation_object_counts, label='Observed', marker='o', color='blue',
        markersize = 10)
# Set ticks at full hours only
ax.set_xticks(full_hours)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax.set_ylim((0, 10))

ax.set_xlabel("Time (UTC)", fontsize = 12)
ax.set_ylabel("Number of convective cores", fontsize = 12)
ax.set_title("Number of convective cores over time", fontsize = 16)
plt.yticks(np.arange(0, 12, 2))
ax.grid(True)
plt.tight_layout()
ax.legend(fontsize = 14)
plt.savefig("../convective_cores_number.png", dpi=300, bbox_inches='tight')
plt.show()





plot_tbmin325_vs_size_by_status(
    filtered_sim_objects,
    bt_325,
    time_list, 
    output_txt = "../tbmin_stats.txt",
    dx_km=1.0)
