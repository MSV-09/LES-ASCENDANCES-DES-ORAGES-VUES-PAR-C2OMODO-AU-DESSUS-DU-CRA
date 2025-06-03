########################################################################
#---------------------------------------------------------------------------
# Importation des modules necessaires aux fonctions utilisees dans le script
#---------------------------------------------------------------------------
import numpy as np
from netCDF4 import Dataset
import matplotlib as mpl
mpl.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from matplotlib import pyplot as plt
from netCDF4 import num2date
import os
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

#-------------------------------
# Adresse et nom de l'expérience
ypath='../Simulations/Sim_3_CRA/'
yexp='FLY15'
filename0 = ypath + yexp + '.1.SEP13.OUT.001.nc'

# List all files in the directory
all_files = os.listdir(ypath)

# Filter for files ending with an uneven number (odd number)
odd_files = [f for f in all_files if f.endswith('.nc') and 
             int(f.split('.')[-2]) % 2  != 0]

# Sort the files based on the number before the '.nc' extension
odd_files.sort(key=lambda f: int(f.split('.')[-2]))


# Altitude maximale de la figure
zaltmax= 12 # in km
ystat = 'point'  # options: point or average or maximum
yvar = 'WT'     # options: "WT"  "UT" "VT" "RT" "THT" "CLDFR"

########################################################################
#FIND CRA COORDINATES
#######################################################################

with Dataset(filename0, 'r') as data:
    lat = data.variables['latitude'][:]  # Latitude array
    lon = data.variables['longitude'][:]  # Longitude array

# Given target latitude and longitude
target_lat = 43.128 # Example: replace with your latitude
target_lon = 0.366  # Example: replace with your longitude


# Calculate the absolute difference in latitudes and longitudes
lat_diff = np.abs(lat - target_lat)
lon_diff = np.abs(lon - target_lon)

# Find the index of the closest point
min_idx = np.argmin(lat_diff + lon_diff)  # Sum the differences to get closest point
ji, ii = np.unravel_index(min_idx, lat.shape)

#print(ji, ii)


########################################################################
#EXTRACT TIME
#######################################################################
# Initialize list to store time data from all odd files
time_list = []

# Loop over the sorted odd-numbered files
for jt, filename in enumerate(odd_files):
    file_path = os.path.join(ypath, filename)
    with Dataset(file_path, 'r') as data:
        # Extract time values and units
        time = data.variables['time'][:]  # Time values (e.g., days since the reference time)
        time_units = data.variables['time'].units  # e.g., "days since 2000-01-01"
        
        # Convert time to datetime objects
        time_dates = num2date(time, units=time_units)
        
        # Append the time data for each file to the list
        time_list.extend(time_dates)

#time_list = [t.strftime('%H:%M') for t in time_list]

# #Only full hours for the plot
# if isinstance(time_list[0], str):
#     time_list = [datetime.strptime(t, '%H:%M') for t in time_list]

# full_hour_indices = [i for i, t in enumerate(time_list) if t.minute == 0]

# full_hour_labels = [t.strftime('%H:%M') for i, t in enumerate(time_list) \
#                     if t.minute == 0]

    # Convert to matplotlib date format
time_list_num = mdates.date2num(time_list)
full_hour_indices = [i for i, t in enumerate(time_list) if t.minute == 0]
full_hour_positions = [time_list_num[i] for i in full_hour_indices]
full_hour_labels = [time_list[i].strftime('%H:%M') for i in full_hour_indices]

    
    
# Caractéristique de la simulation
ntime=len(odd_files) # Nombre de fichiers en sortie de la simulation
nlevel=113 # Nombre de niveaux verticaux
########################################################################
#------------------------------
# Lecture du fichier de donnees
#------------------------------

# Tableaux vides
WT = np.full((nlevel, ntime), np.nan)
UT = np.full((nlevel, ntime), np.nan)
VT = np.full((nlevel, ntime), np.nan)
RT_3 = np.full((nlevel, ntime), np.nan)
RT_5 = np.full((nlevel, ntime), np.nan)
THT = np.full((nlevel, ntime), np.nan)
CLDFR = np.full((nlevel, ntime), np.nan)
RT = np.full((nlevel, ntime), np.nan)

ztime = np.zeros([ntime], dtype=float)

# Load vertical levels once

with Dataset(filename0, 'r') as data:
    altitude = data.variables['level'][:] * 0.001  # in km

# Loop over the filtered files
for jt, filename in enumerate(odd_files):
    file_path = os.path.join(ypath, filename)
    

    if ystat=='point':
        with Dataset(file_path, 'r') as data:
            if yvar=='WT':
                WT[:, jt] = data.variables['WT'][0, :, ji, ii]  
                RT_3[:, jt] = data.variables['RT'][0, 3, :, ji, ii]  *1000      
                RT_5[:, jt] = data.variables['RT'][0, 5, :, ji, ii]  *1000    
                
            if yvar=='UT':
                UT[:, jt] = data.variables['UT'][0, :, ji, ii]  
                
            if yvar=='VT':
                VT[:, jt] = data.variables['VT'][0, :, ji, ii]  
                
            if yvar=='THT':
                THT[:, jt] = data.variables['THT'][0, :, ji, ii]  
                
            if yvar=='CLDFR':
                CLDFR[:, jt] = data.variables['CLDFR'][0, :, ji, ii]  
                
            if yvar=='RT':
                RT[:, jt] = data.variables['RT'][0, 0, :, ji, ii]*1000 
                WT[:, jt] = data.variables['WT'][0, :, ji, ii]
                
            
    if ystat=='average':
        with Dataset(file_path, 'r') as data:
            if yvar=='WT':
                WT[:, jt] = np.mean(data.variables['WT'][0, :, ji, ii])
                
            if yvar=='UT':
                UT[:, jt] = np.mean(data.variables['UT'][0, :, ji, ii])
                
            if yvar=='VT':
                VT[:, jt] = np.mean(data.variables['VT'][0, :, ji, ii])  
                
            if yvar=='THT':
                THT[:, jt] = np.mean(data.variables['THT'][0, :, ji, ii]) 
                
            if yvar=='CLDFR':
                CLDFR[:, jt] = np.mean(data.variables['CLDFR'][0, :, ji, ii])  
                
            if yvar=='RT':
                RT[:, jt] = np.mean(data.variables['RT'][0, 0, :, ji, ii])  
                
                
    elif ystat=='maximum':
        with Dataset(file_path, 'r') as data:
            if yvar=='WT':
                WT[:, jt] = np.amax(data.variables['WT'][0, :, ji, ii])
                
            if yvar=='UT':
                UT[:, jt] = np.amax(data.variables['UT'][0, :, ji, ii])
                
            if yvar=='VT':
                VT[:, jt] = np.amax(data.variables['VT'][0, :, ji, ii])  
                
            if yvar=='THT':
                THT[:, jt] = np.amax(data.variables['THT'][0, :, ji, ii]) 
                
            if yvar=='CLDFR':
                CLDFR[:, jt] = np.amax(data.variables['CLDFR'][0, :, ji, ii])  
                
            if yvar=='RT':
                RT[:, jt] = np.amax(data.variables['RT'][0, 0, :, ji, ii]) 

#########################################################################
# Parametre du tracer de la figure
#---------------------------------

# Save the figure as PNG and also show it in interactive environments
plt.figure(figsize=(10, 6))

# if yvar=='WT':
#     plt.imshow(WT, aspect='auto', origin='lower',
#                extent=[0, len(time_list), altitude[0], altitude[-1]],
#                cmap='jet')
#     # Set the x-axis to correspond to the time values
#     plt.xticks(ticks = full_hour_indices, labels = full_hour_labels, rotation=45)
#     # Add colorbar and labels
#     plt.colorbar(label='Vertical Velocity (m s^-1)')
#     plt.xlabel('Time UTC')
#     plt.ylabel('Altitude (km)')
#     plt.ylim((0, 10))
#     plt.grid(True)
#     plt.title('Time-depth plot of Vertical Wind Velocity CRA on 05/29/2023')
#     plt.savefig('WT' + '_' + ystat +'.png', dpi=300, bbox_inches='tight')
#     # Make layout tight to avoid overlap
#     plt.tight_layout()

if yvar=='UT':
    plt.imshow(UT, aspect='auto', origin='lower',
               extent=[0, len(time_list), altitude[0], altitude[-1]],
               cmap='jet')
    # Set the x-axis to correspond to the time values
    plt.xticks(ticks = full_hour_indices, labels = full_hour_labels, rotation=45)
    # Add colorbar and labels
    plt.colorbar(label='Horizontal Velocity (U) (m s^-1)')
    plt.xlabel('Time UTC')
    plt.ylabel('Altitude (km)')
    plt.ylim((0, 10))
    plt.grid(True)
    plt.title('Time-depth plot of Horizontal Wind Velocity CRA on 05/29/2023')
    plt.savefig('UT' + '_' + ystat +'.png', dpi=300, bbox_inches='tight')
    # Make layout tight to avoid overlap
    plt.tight_layout()

if yvar=='VT':
    plt.imshow(VT, aspect='auto', origin='lower',
               extent=[0, len(time_list), altitude[0], altitude[-1]],
               cmap='jet')
    # Set the x-axis to correspond to the time values
    plt.xticks(ticks = full_hour_indices, labels = full_hour_labels, rotation=45)
    # Add colorbar and labels
    plt.colorbar(label='Horizontal Velocity (V) (m s^-1)')
    plt.xlabel('Time UTC')
    plt.ylabel('Altitude (km)')
    plt.ylim((0, 10))
    plt.grid(True)
    plt.title('Time-depth plot of Horizontal Wind Velocity CRA on 05/29/2023')
    plt.savefig('VT' + '_' + ystat +'.png', dpi=300, bbox_inches='tight')
    # Make layout tight to avoid overlap
    plt.tight_layout()

if yvar=='THT':
    plt.imshow(THT, aspect='auto', origin='lower',
               extent=[0, len(time_list), altitude[0], altitude[-1]],
               cmap='jet')
    # Set the x-axis to correspond to the time values
    plt.xticks(ticks = full_hour_indices, labels = full_hour_labels, rotation=45)
    # Add colorbar and labels
    plt.colorbar(label='THT')
    plt.xlabel('Time UTC')
    plt.ylabel('Altitude (km)')
    plt.ylim((0, 10))
    plt.grid(True)
    plt.title('Time-depth plot of THT above CRA on 05/29/2023')
    plt.savefig('THT' + '_' + ystat +'.png', dpi=300, bbox_inches='tight')
    # Make layout tight to avoid overlap
    plt.tight_layout()

if yvar=='CLDFR':
    plt.imshow(CLDFR, aspect='auto', origin='lower',
               extent=[0, len(time_list), altitude[0], altitude[-1]],
               cmap='jet')
    # Set the x-axis to correspond to the time values
    plt.xticks(ticks = full_hour_indices, labels = full_hour_labels, rotation=45)
    # Add colorbar and labels
    plt.colorbar(label='CLDFR')
    plt.xlabel('Time UTC')
    plt.ylabel('Altitude (km)')
    plt.ylim((0, 10))
    plt.grid(True)
    plt.title('Time-depth plot of CLDFR above CRA on 05/29/2023')
    plt.savefig('CLDFR' + '_' + ystat +'.png', dpi=300, bbox_inches='tight')
    # Make layout tight to avoid overlap
    plt.tight_layout()


if yvar == 'RT':
    fig, ax = plt.subplots(figsize=(10, 6))

    # Affichage de RT
    im = ax.imshow(RT, aspect='auto', origin='lower',
                   extent=[time_list_num[0], time_list_num[-1], altitude[0], altitude[-1]],
                   cmap='jet')

    level = np.arange(-3, 3, 1)

    # Ajout des contours de WT
    CS = ax.contour(time_list_num, altitude, WT, 
                    levels=level,  # ou choisis manuellement ex: levels=[-1, 0, 1, 2, 3]
                    colors='k', linewidths=1)

    ax.clabel(CS, inline=True, fontsize=8, fmt='%1.1f')  # Étiquettes sur les courbes

    # Mise en forme
    ax.set_ylabel('Altitude (km)')
    ax.set_xlabel('Time UTC')
    ax.set_ylim((0, 10))    
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_title('Time-depth plot of RT above CRA on 05/29/2023')
    ax.grid(True)

    # Barre de couleur
    plt.colorbar(im, ax=ax, label='Rapport de Mélange (kg kg$^{{-1}}$)')

    plt.tight_layout()
    plt.savefig('RT_' + ystat + '.png', dpi=300, bbox_inches='tight')
    plt.show()

if yvar == 'WT':
    fig, ax = plt.subplots(figsize=(10, 6))

    # Affichage de RT
    im = ax.imshow(WT, aspect='auto', origin='lower',
                   extent=[time_list_num[0], time_list_num[-1], altitude[0], altitude[-1]],
                   cmap='jet', vmin=-3, vmax=3)

    T, Z = np.meshgrid(time_list_num, altitude)  # Grille régulière
    CS_3 = ax.contour(T, Z, RT_3, levels=[0.01], colors='k',  linewidths=1, linestyles = '--')
    CS_5 = ax.contour(T, Z, RT_5, levels=[0.5, 1.0, 2.0, 3.0], colors='k', linewidths=1)

           
    ax.clabel(CS_3, inline=True, fontsize=8, fmt='%1.1f' )
    ax.clabel(CS_5, inline=True, fontsize=8, fmt='%1.1f' )

    ax.axhline(y=12, color='saddlebrown', linestyle='--', linewidth=1.5, label='Limite Troposphère')

    legend_lines = [
        Line2D([0], [0], color='k', linewidth=1, linestyle = '--', label='Glace nuageuse'),
        Line2D([0], [0], color='k', linewidth=1, label='Graupel'),
        Line2D([0], [0], color='saddlebrown', linestyle='--', linewidth=1.5, label='Tropopause')
        ]

    ax.legend(handles = legend_lines, loc='upper right', bbox_to_anchor=(1, 0.95), fontsize=12)

    # Mise en forme
    ax.set_ylabel('Altitude (km)', fontsize=14)
    ax.set_xlabel('Heure UTC', fontsize=14)
    ax.set_ylim((0, 15))    
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_title('Time-depth plot of WT above CRA on 05/29/2023')
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True)

    # Personnalisation de la colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Vitesse verticale (m s$^{-1}$)', fontsize=14)
    cbar.ax.tick_params(labelsize=14)  # Taille des ticks si besoin

    plt.tight_layout()
    plt.savefig('WT_' + ystat + '.png', dpi=300, bbox_inches='tight')
    plt.show()


