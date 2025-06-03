import numpy as np
import cartopy.crs as ccrs
from collections import defaultdict
import matplotlib.cm as cm
import random
import matplotlib.colors as mcolors
from collections import Counter
from scipy.stats import linregress


from scipy.stats import shapiro, f_oneway, spearmanr, linregress
import scipy.stats as st
from scipy.stats import shapiro, levene, spearmanr, pearsonr, f
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import pandas as pd




def filter_objects_by_size(objects, min_pixels=20, mode='per_timestep'):
    """
    Filters labeled objects in a 3D array (time, y, x) by object size.
    
    Parameters:
    -----------
    objects : np.ndarray
        3D array of labeled objects. Background = 0.
    min_pixels : int
        Minimum number of pixels required.
    mode : str
        - 'total': filters by total object size across all time steps.
        - 'per_timestep': filters by size at each timestep individually.
    
    Returns:
    --------
    filtered_objects : np.ndarray
        Same shape as `objects`, with small objects removed (set to 0).
    """
    filtered_objects = np.copy(objects)
    
    if mode == 'total':
        pixel_counts = defaultdict(int)
        for t in range(objects.shape[0]):
            unique_ids, counts = np.unique(objects[t], return_counts=True)
            for obj_id, count in zip(unique_ids, counts):
                if obj_id != 0:
                    pixel_counts[obj_id] += count
        valid_ids = {obj_id for obj_id, count in pixel_counts.items() if count >= min_pixels}
        filtered_objects = np.where(np.isin(objects, list(valid_ids)), objects, 0)

    elif mode == 'per_timestep':
        for t in range(objects.shape[0]):
            frame = filtered_objects[t]
            labels, counts = np.unique(frame, return_counts=True)
            small_labels = labels[(counts < min_pixels) & (labels != 0)]
            mask = np.isin(frame, small_labels)
            filtered_objects[t][mask] = 0

    else:
        raise ValueError("Mode must be either 'total' or 'per_timestep'.")

    return filtered_objects




def generate_consistent_colors(unique_ids):
    """
    Generate a consistent set of colors for the given unique object IDs.
    Uses a fixed seed for reproducibility.
    """
    random.seed(42)  # Fixed seed to ensure the same colors across timesteps
    colors = [(0, 0, 0, 0)]  # Start with transparent for background (ID 0)
    for _ in range(len(unique_ids) - 1):  # We already have the background color
        colors.append((random.random(), random.random(), random.random(), 1.0))
    
    return mcolors.ListedColormap(colors)

def plot_object_mask_pcolormesh(ax, objects_mask, lon, lat, zorder=10):
    """
    Plots labeled objects using pcolormesh on a geographic map.

    Parameters:
        ax : matplotlib Axes
            The axis with a Cartopy projection to plot on.
        objects_mask : 2D np.array
            Labeled object mask (e.g., from a segmentation algorithm).
        lon : 2D np.array or 1D np.array
            Longitude array matching the shape of objects_mask.
        lat : 2D np.array or 1D np.array
            Latitude array matching the shape of objects_mask.
        zorder : int
            Drawing order of the pcolormesh.
    """

    # Step 1: Ensure lon and lat are 2D arrays if they are 1D
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)  # Create meshgrid if lon/lat are 1D
    
    # Step 2: Identify unique object IDs (excluding background = 0)
    unique_ids = np.unique(objects_mask)
    unique_ids = unique_ids[unique_ids != 0]  # remove background (ID=0)

    # Step 3: Map object IDs to sequential color indices
    id_to_index = {obj_id: i + 1 for i, obj_id in enumerate(unique_ids)}  # start at 1
    objects_indexed = np.zeros_like(objects_mask, dtype=int)
    for obj_id, idx in id_to_index.items():
        objects_indexed[objects_mask == obj_id] = idx

    # Step 4: Generate consistent colormap for all timesteps
    rand_cmap = generate_consistent_colors(unique_ids)

    # Ensure lon/lat match the shape of objects_mask
    lon = lon[:objects_mask.shape[0] + 1, :objects_mask.shape[1] + 1]
    lat = lat[:objects_mask.shape[0] + 1, :objects_mask.shape[1] + 1]

    # Step 5: Plot using pcolormesh
    mesh = ax.pcolormesh(
        lon, lat, objects_indexed,
        cmap=rand_cmap, shading='auto',
        transform=ccrs.PlateCarree(), zorder=zorder
    )

    return mesh  # for optional use (e.g., colorbar)


def count_objects(objects_mask):
    """
    Count the number of objects (unique IDs) in the objects mask.
    Excludes background (ID = 0).

    Parameters:
        objects_mask : 2D np.array
            Labeled object mask (e.g., from a segmentation algorithm).

    Returns:
        num_objects : int
            The number of unique objects (excluding background).
    """
    # Identify unique object IDs (excluding background = 0)
    unique_ids = np.unique(objects_mask)
    unique_ids = unique_ids[unique_ids != 0]  # Remove background (ID=0)

    # Count the number of unique IDs
    num_objects = len(unique_ids)
    
    return num_objects



def get_object_sizes(label_array):
    """
    Returns a dictionary {object_id: size (in pixels)} for a labeled array.
    """
    labels = label_array[label_array != 0]  # exclude background
    return dict(Counter(labels))




def plot_tbmin_vs_size_by_status(label_3d_sim, label_3d_obs, tb_sim, tb_obs, timesteps, output_txt, dx_km=3.0):
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter

    def get_object_sizes(label_array):
        """
        Returns a dictionary {object_id: size (in pixels)} for a labeled array.
        """
        labels = label_array[label_array != 0]  # exclude background
        return dict(Counter(labels))
    
    def extract_sizes_and_min_tb_with_status(label_3d, tb_3d, status_map, dx_km=3.0):
        sizes_km2, min_tb_values, status_values = [], [], []
        for t in range(label_3d.shape[0]):
            labels = label_3d[t]
            tb = tb_3d[t]  # shape: (y, x)
            object_sizes = get_object_sizes(labels)
    
            for obj_id, size in object_sizes.items():
                if obj_id == 0:
                    continue  # Skip background
    
                coords = np.column_stack(np.where(labels == obj_id))
                if coords.size == 0:
                    continue
    
                try:
                    # Clip coordinates to ensure they're within TB array bounds
                    coords = np.clip(coords, [0, 0], [tb.shape[0]-1, tb.shape[1]-1])
    
                    # Extract TB values for object pixels
                    tb_values = tb[coords[:, 0], coords[:, 1]]
                    min_tb = np.nanmin(tb_values)
    
                    # Calculate area in km²
                    area_km2 = size * dx_km**2
    
                    # Status from the map using first pixel
                    status = status_map[t][coords[0][0], coords[0][1]]
    
                    # Store values
                    sizes_km2.append(area_km2)
                    min_tb_values.append(min_tb)
                    status_values.append(status)
    
                except Exception as e:
                    print(f"⚠️ Error at timestep {t}, object {obj_id}: {e}")
        return sizes_km2, min_tb_values, status_values



    # === Compute status maps ===
    print("📦 Classifying object evolution for SIM...")
    status_map_sim = classify_object_evolution(label_3d_sim)
    print("📦 Classifying object evolution for OBS...")
    status_map_obs = classify_object_evolution(label_3d_obs)

    # === Extract data ===
    print("\n🔍 Extracting data for SIM...")
    sizes_sim, tbmins_sim, status_sim = extract_sizes_and_min_tb_with_status(label_3d_sim, tb_sim, status_map_sim)
    print("\n🔍 Extracting data for OBS...")
    sizes_obs, tbmins_obs, status_obs = extract_sizes_and_min_tb_with_status(label_3d_obs, tb_obs, status_map_obs)

    if not sizes_sim and not sizes_obs:
        print("⚠️ No valid data to plot.")
        return
    
        
    # === Debugging print before plotting ===
    print(f"Simulation sizes: {sizes_sim[:10]}...")  # Print the first 10 for brevity
    print(f"Observation sizes: {sizes_obs[:10]}...")  # Print the first 10 for brevity
    
    # === Check for zero or negative values ===
    if any(s <= 0 for s in sizes_sim):
        print(f"⚠️ Warning: Some simulation sizes are zero or negative.")
    if any(s <= 0 for s in sizes_obs):
        print(f"⚠️ Warning: Some observation sizes are zero or negative.")
        
        
    log_sizes_sim = np.array(np.log10(sizes_sim)) 
    log_sizes_obs = np.array(np.log10(sizes_obs)) 
    
    slope_sim, intercept_sim, r_value_sim, p_value_sim, std_err_sim = linregress(log_sizes_sim, np.array(tbmins_sim))  
    slope_obs, intercept_obs, r_value_obs, p_value_obs, std_err_obs = linregress(log_sizes_obs, np.array(tbmins_obs))
    
    
    # Regression prediction on actual data
    predicted_on_data_sim = slope_sim * log_sizes_sim + intercept_sim
    residuals_sim = np.array(tbmins_sim) - predicted_on_data_sim
    sigma_sim = np.std(residuals_sim)  # 1σ based on actual fit
    
    predicted_on_data_obs = slope_obs * log_sizes_obs + intercept_obs
    residuals_obs = np.array(tbmins_obs) - predicted_on_data_obs
    sigma_obs = np.std(residuals_obs)  # 1σ based on actual fit

    # Regression line
    x_line_sim = np.logspace(np.log10(min(sizes_sim)), np.log10(max(sizes_sim)), 100)
    y_line_sim = slope_sim * np.log10(x_line_sim) + intercept_sim
    upper_line_sim = y_line_sim + 1.96*sigma_sim
    lower_line_sim = y_line_sim - 1.96*sigma_sim
    
    x_line_obs = np.logspace(np.log10(min(sizes_obs)), np.log10(max(sizes_obs)), 100)
    y_line_obs = slope_obs * np.log10(x_line_obs) + intercept_obs
    upper_line_obs = y_line_obs + 1.96*sigma_obs
    lower_line_obs = y_line_obs - 1.96*sigma_obs
    


    # === Plotting settings ===
    status_labels = {1: "En croissance", -1: "En dissipation", 0: "Stable"}
    status_colors = {1: 'g', -1: 'r', 0: 'b'}
    

    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    for status in [-1, 0, 1]:  # Decaying, Stable, Growing
        indices_sim = [i for i, s in enumerate(status_sim) if s == status]
        if indices_sim:
            ax[1].scatter(
                np.array(sizes_sim)[indices_sim],
                np.array(tbmins_sim)[indices_sim],
                color=status_colors[status],
                label=status_labels[status],
                alpha=0.7,
                s=50
            )

        indices_obs = [i for i, s in enumerate(status_obs) if s == status]
        if indices_obs:
            ax[0].scatter(
                np.array(sizes_obs)[indices_obs],
                np.array(tbmins_obs)[indices_obs],
                color=status_colors[status],
                label=status_labels[status],
                alpha=0.7,
                s=50
            )


    ax[1].plot(x_line_sim, y_line_sim,
            color='black', linestyle='-', linewidth=2,
            label=(f"Régression linéaire\n"
                f"$y = {slope_sim:.1f} \cdot \log_{{10}}(x) + {intercept_sim:.1f}$\n"
                f"$R^2 = {r_value_sim**2:.2f}$"
            )
        )
        
    ax[1].fill_between(x_line_sim, lower_line_sim, upper_line_sim, color='black', alpha=0.1)
    ax[1].set_title("Simulation (Méso-NH)", fontsize = 18)
    ax[1].set_xlabel("Surface de la cellule (km²)", fontsize = 16)
    ax[1].set_ylabel("$T_{b,\\ min}$ (K)", fontsize = 18)
    ax[1].set_xscale('log')
    ax[1].grid(True)
    ax[1].tick_params(axis='both', labelsize = 16)

    ax[0].plot(x_line_obs, y_line_obs,
               color='black', linestyle='-', linewidth=2,
               label=(f"Régression linéaire\n"
                    f"$y = {slope_obs:.1f} \cdot \log_{{10}}(x) + {intercept_obs:.1f}$\n"
                    f"$R^2 = {r_value_obs**2:.2f}$"
                )
            )
            
    ax[0].fill_between(x_line_obs, lower_line_obs, upper_line_obs, color='black', alpha=0.1)


    ax[0].set_title("Observation (MSG)", fontsize = 18)
    ax[0].set_xlabel("Surface de l'objet (km²)", fontsize = 16)
    ax[0].set_ylabel("$T_{b,\\ min}$ (K)", fontsize = 18)
    ax[0].set_xscale('log')
    ax[0].grid(True)
    ax[0].tick_params(axis='both', labelsize = 16)
    ax[0].legend(fontsize = 14, loc = 3)
    ax[1].legend(fontsize = 14, loc = 3)
    plt.tight_layout()
    plt.savefig("../Min_Tb_vs_objects_size_by_status.png", dpi=300, bbox_inches="tight")
    plt.show()


    
    
def classify_object_evolution(objects_3d, dx_km=1.0, threshold_pct=10):
    from collections import defaultdict
    import numpy as np

    n_times = objects_3d.shape[0]
    object_size_history = defaultdict(lambda: np.zeros(n_times))

    # Step 1: Record object size per timestep
    for t in range(n_times):
        labels = objects_3d[t]
        unique_ids = np.unique(labels)
        for obj_id in unique_ids:
            if obj_id == 0:
                continue
            size = np.sum(labels == obj_id)
            object_size_history[obj_id][t] = size * dx_km**2  # convert to km²

    # Step 2: Classify evolution
    status_map = np.zeros_like(objects_3d, dtype=np.int8)

    print("Classifying object evolution...\n")
    
    # For each object, calculate its status at each timestep
    for obj_id, sizes in object_size_history.items():
        print(f"Object {obj_id}:")
        
        for t in range(1, n_times - 1):
            prev, curr, nxt = sizes[t - 1], sizes[t], sizes[t + 1]
            pct_change = lambda a, b: 100 * (b - a) / a if a > 0 else 0
            
            if pct_change(prev, curr) > threshold_pct and pct_change(curr, nxt) > threshold_pct:
                status = 1  # growing
            elif pct_change(prev, curr) < -threshold_pct and pct_change(curr, nxt) < -threshold_pct:
                status = -1 # decaying
            else:
                status = 0  # stable

            mask = (objects_3d[t] == obj_id)
            status_map[t][mask] = status

            # Print the status of the object at each timestep
            status_str = {1: "Growing", -1: "Decaying", 0: "Stable"}[status]
            print(f"  Timestep {t + 1}: {status_str} (Size: {sizes[t]:.2f} km²)")

        print("\n")

    return status_map





def plot_tbmin325_vs_size_by_status(label_3d_sim, tb_sim, timesteps, output_txt = "../tbmin_stats.txt", dx_km=1.0):
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter

    def get_object_sizes(label_array):
        """
        Returns a dictionary {object_id: size (in pixels)} for a labeled array.
        """
        labels = label_array[label_array != 0]  # exclude background
        return dict(Counter(labels))
    
    def extract_sizes_and_min_tb_with_status(label_3d, tb_3d, status_map, dx_km=3.0):
        sizes_km2, min_tb_values, status_values = [], [], []
        for t in range(label_3d.shape[0]):
            labels = label_3d[t]
            tb = tb_3d[t]  # shape: (y, x)
            object_sizes = get_object_sizes(labels)
    
            for obj_id, size in object_sizes.items():
                if obj_id == 0:
                    continue  # Skip background
    
                coords = np.column_stack(np.where(labels == obj_id))
                if coords.size == 0:
                    continue
    
                try:
                    # Clip coordinates to ensure they're within TB array bounds
                    coords = np.clip(coords, [0, 0], [tb.shape[0]-1, tb.shape[1]-1])
    
                    # Extract TB values for object pixels
                    tb_values = tb[coords[:, 0], coords[:, 1]]
                    min_tb = np.nanmin(tb_values)
    
                    # Calculate area in km²
                    area_km2 = size * dx_km**2
    
                    # Status from the map using first pixel
                    status = status_map[t][coords[0][0], coords[0][1]]
    
                    # Store values
                    sizes_km2.append(area_km2)
                    min_tb_values.append(min_tb)
                    status_values.append(status)
    
                except Exception as e:
                    print(f"⚠️ Error at timestep {t}, object {obj_id}: {e}")
        return sizes_km2, min_tb_values, status_values



    # === Compute status maps ===
    status_map_sim = classify_object_evolution(label_3d_sim)


    # === Extract data ===
    sizes_sim, tbmins_sim, status_sim = extract_sizes_and_min_tb_with_status(label_3d_sim, tb_sim, status_map_sim)      
        
    log_sizes_sim = np.array(np.log10(sizes_sim)) 
    
    slope_sim, intercept_sim, r_value_sim, p_value_sim, std_err_sim = linregress(log_sizes_sim, np.array(tbmins_sim))  

    
    # Regression prediction on actual data
    predicted_on_data_sim = slope_sim * log_sizes_sim + intercept_sim
    residuals_sim = np.array(tbmins_sim) - predicted_on_data_sim
    sigma_sim = np.std(residuals_sim)  # 1σ based on actual fit
    


    # Regression line
    x_line_sim = np.logspace(np.log10(min(sizes_sim)), np.log10(max(sizes_sim)), 100)
    y_line_sim = slope_sim * np.log10(x_line_sim) + intercept_sim
    upper_line_sim = y_line_sim + 1.96*sigma_sim
    lower_line_sim = y_line_sim - 1.96*sigma_sim


    # === Plotting settings ===
    status_labels = {1: "En croissance", -1: "En dissipation", 0: "Stable"}
    status_colors = {1: 'g', -1: 'r', 0: 'b'}
    

    fig, ax = plt.subplots(figsize=(10, 8))

    for status in [-1, 0, 1]:  # Decaying, Stable, Growing
        indices_sim = [i for i, s in enumerate(status_sim) if s == status]
        if indices_sim:
            ax.scatter(
                np.array(sizes_sim)[indices_sim],
                np.array(tbmins_sim)[indices_sim],
                color=status_colors[status],
                label=status_labels[status],
                alpha=0.7,
                s=50
            )



    ax.plot(x_line_sim, y_line_sim,
            color='black', linestyle='-', linewidth=2,
            label=(f"Régression linéaire\n"
                f"$y = {slope_sim:.1f} \cdot \log_{{10}}(x) + {intercept_sim:.1f}$\n"
                f"$R^2 = {r_value_sim**2:.2f}$"
            )
        )
        
    ax.fill_between(x_line_sim, lower_line_sim, upper_line_sim, color='black', alpha=0.1)
    ax.set_xlabel("Surface de la cellule (km²)", fontsize = 18)
    ax.set_ylabel("$T_{b,\\ min}$ à 325 GHz (K)", fontsize = 18)
    ax.set_xscale('log')
    ax.grid(True)
    ax.tick_params(axis='both', labelsize = 16)

    ax.legend(fontsize = 14, loc = 3)
    plt.tight_layout()
    #plt.savefig("../Min_Tb_vs_objects_size_by_status.png", dpi=300, bbox_inches="tight")
    plt.show()

    def write_stats_by_status(sizes_km2, min_tb_values, status_values, output_txt):

        stades = [1, 0, -1]

    

        # Créer un DataFrame à partir des listes

        df = pd.DataFrame({

            'Size': sizes_km2,

            'WT_max': min_tb_values,  # Ici Tb_min = WT_max dans ton contexte

            'Status': status_values

        })

    

        with open(output_txt, 'w') as f:

            f.write("=== Statistiques par stade de cellule ===\n\n")

    

            # Vérification des colonnes disponibles

            print("Colonnes du DataFrame :", df.columns.tolist())

            if 'Status' not in df.columns or 'WT_max' not in df.columns or 'Size' not in df.columns:

                raise ValueError(

                    "Le DataFrame doit contenir les colonnes : 'Status', 'WT_max', 'Size'")

    

            print("Valeurs uniques dans 'Status' :", df['Status'].unique())

    

            for stade in stades:

                subset = df[df['Status'] == stade]

    

                if subset.empty:

                    f.write(

                        f"--- {stade} ---\nAucune donnée disponible pour ce stade.\n\n")

                    continue

    

                wt = subset['WT_max'].dropna().values

                sizes = subset['Size'].dropna().values

    

                if len(wt) != len(sizes) or len(wt) == 0:

                    f.write(

                        f"--- {stade} ---\nDonnées insuffisantes ou incomplètes pour calculer les statistiques.\n\n")

                    continue

    

                log_sizes = np.log10(sizes)

    

                # Régression linéaire

                slope, intercept, r_value, p_val, std_err = st.linregress(

                    log_sizes, wt)

                r2 = r_value ** 2

    

                # Moyenne / écart-type

                mean_wt = np.mean(wt)

                std_wt = np.std(wt, ddof=1)

    

                # Skewness et Shapiro

                skew = st.skew(wt)

                shapiro_stat, shapiro_p = shapiro(wt)

    

                # Spearman

                spearman_corr, spearman_p = spearmanr(log_sizes, wt)

    

                f.write(f"--- {stade} ---\n")

                f.write(f"n                         : {len(wt)}\n")

                f.write(f"Moyenne Tb_min           : {mean_wt:.3f}\n")

                f.write(f"Écart-type Tb_min        : {std_wt:.3f}\n")

                f.write(f"Skewness                 : {skew:.3f}\n")

                f.write(           f"Shapiro-Wilk W           : {shapiro_stat:.3f} (p = {shapiro_p:.3f})\n")

                f.write(f"Régression : Intercept   : {intercept:.3f}\n")

                f.write(f"              R²         : {r2:.3f}\n")

                f.write(

                    f"Spearman ρ               : {spearman_corr:.3f} (p = {spearman_p:.3f})\n\n")



    write_stats_by_status(sizes_sim, tbmins_sim, status_sim, output_txt)
    
    
def classify_object_evolution(objects_3d, dx_km=1.0, threshold_pct=10):
    from collections import defaultdict
    import numpy as np

    n_times = objects_3d.shape[0]
    object_size_history = defaultdict(lambda: np.zeros(n_times))

    # Step 1: Record object size per timestep
    for t in range(n_times):
        labels = objects_3d[t]
        unique_ids = np.unique(labels)
        for obj_id in unique_ids:
            if obj_id == 0:
                continue
            size = np.sum(labels == obj_id)
            object_size_history[obj_id][t] = size * dx_km**2  # convert to km²

    # Step 2: Classify evolution
    status_map = np.zeros_like(objects_3d, dtype=np.int8)

    print("Classifying object evolution...\n")
    
    # For each object, calculate its status at each timestep
    for obj_id, sizes in object_size_history.items():
        print(f"Object {obj_id}:")
        
        for t in range(1, n_times - 1):
            prev, curr, nxt = sizes[t - 1], sizes[t], sizes[t + 1]
            pct_change = lambda a, b: 100 * (b - a) / a if a > 0 else 0
            
            if pct_change(prev, curr) > threshold_pct and pct_change(curr, nxt) > threshold_pct:
                status = 1  # growing
            elif pct_change(prev, curr) < -threshold_pct and pct_change(curr, nxt) < -threshold_pct:
                status = -1 # decaying
            else:
                status = 0  # stable

            mask = (objects_3d[t] == obj_id)
            status_map[t][mask] = status

            # Print the status of the object at each timestep
            status_str = {1: "Growing", -1: "Decaying", 0: "Stable"}[status]
            print(f"  Timestep {t + 1}: {status_str} (Size: {sizes[t]:.2f} km²)")

        print("\n")

    return status_map



def write_stats_by_status(sizes_km2, min_tb_values, status_values, output_txt):
    stades = [1, 0, -1]

    # CrÃ©er un DataFrame Ã  partir des listes
    df = pd.DataFrame({
        'Size': sizes_km2,
        'WT_max': min_tb_values,  # Ici Tb_min = WT_max dans ton contexte
        'Status': status_values
    })

    with open(output_txt, 'w') as f:
        f.write("=== Statistiques par stade de cellule ===\n\n")

        # VÃ©rification des colonnes disponibles
        print("Colonnes du DataFrame :", df.columns.tolist())
        if 'Status' not in df.columns or 'WT_max' not in df.columns or 'Size' not in df.columns:
            raise ValueError(
                "Le DataFrame doit contenir les colonnes : 'Status', 'WT_max', 'Size'")

        print("Valeurs uniques dans 'Status' :", df['Status'].unique())

        for stade in stades:
            subset = df[df['Status'] == stade]

            if subset.empty:
                f.write(
                    f"--- {stade} ---\nAucune donnÃ©e disponible pour ce stade.\n\n")
                continue

            wt = subset['WT_max'].dropna().values
            sizes = subset['Size'].dropna().values

            if len(wt) != len(sizes) or len(wt) == 0:
                f.write(
                    f"--- {stade} ---\nDonnÃ©es insuffisantes ou incomplÃ¨tes pour calculer les statistiques.\n\n")
                continue

            log_sizes = np.log10(sizes)

            # RÃ©gression linÃ©aire
            slope, intercept, r_value, p_val, std_err = st.linregress(
                log_sizes, wt)
            r2 = r_value ** 2

            # Moyenne / Ã©cart-type
            mean_wt = np.mean(wt)
            std_wt = np.std(wt, ddof=1)

            # Skewness et Shapiro
            skew = st.skew(wt)
            shapiro_stat, shapiro_p = shapiro(wt)

            # Spearman
            spearman_corr, spearman_p = spearmanr(log_sizes, wt)

            f.write(f"--- {stade} ---\n")
            f.write(f"n                         : {len(wt)}\n")
            f.write(f"Moyenne Tb_min           : {mean_wt:.3f}\n")
            f.write(f"Ã‰cart-type Tb_min        : {std_wt:.3f}\n")
            f.write(f"Skewness                 : {skew:.3f}\n")
            f.write(           f"Shapiro-Wilk W           : {shapiro_stat:.3f} (p = {shapiro_p:.3f})\n")
            f.write(f"RÃ©gression : Intercept   : {intercept:.3f}\n")
            f.write(f"              RÂ²         : {r2:.3f}\n")
            f.write(
                f"Spearman Ï               : {spearman_corr:.3f} (p = {spearman_p:.3f})\n\n")

write_stats_by_status(sizes_sim, tbmins_sim, status_sim, output_txt)
