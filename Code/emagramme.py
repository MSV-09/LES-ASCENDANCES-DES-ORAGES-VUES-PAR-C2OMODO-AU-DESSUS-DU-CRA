import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
import os
import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units
from datetime import datetime, timedelta
import re
import csv
from metpy.calc import wet_bulb_temperature


# Dossier contenant les fichiers NetCDF
sim_path = '../Simulations/Sim_3_CRA/'
output_folder = 'Emagramme_1/'
os.makedirs(output_folder, exist_ok=True)


# Coordonnées cibles
target_lat = 43.128
target_lon = 0.366
indici = 256
indicj = 256


# === Sélection des fichiers OUT.xxx.nc impairs de 1 à 93 ===
sim_files = os.listdir(sim_path)
sim_pattern = re.compile(r'FLY15.1.SEP13.OUT\.(\d+)\.nc')


filtered_files = []
for f in sim_files:
    match = sim_pattern.match(f)
    if match:
        num = int(match.group(1))
        if 1 <= num <= 72 and num % 2 != 0:
            filtered_files.append(f)

# Trier les fichiers par numéro
filtered_files.sort(key=lambda f: int(sim_pattern.match(f).group(1)))


# === Initialiser les listes CAPE/CIN ===
cape_values = []
cin_values = []
timesteps = []      # labels or timestamps
i=0

# Step 1: Create the list of times from 14:00:00 to 17:45:00 with 15-min steps
start = datetime.strptime("13-15-00", "%H-%M-%S")
end = datetime.strptime("22-00-00", "%H-%M-%S")
time_list = []

current_time = start
while current_time <= end:
    time_str = current_time.strftime("%H-%M-%S")
    time_list.append(time_str)
    current_time += timedelta(minutes=15)

vertical_profile_csv = os.path.join(output_folder, 'profils_verticaux.csv')

with open(vertical_profile_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Heure', 'Niveau', 'Pression (hPa)', 'Température (°C)', 'Theta (K)', 'Theta_w (K)'])

# Boucle sur tous les fichiers .nc dans le dossier
for i, filename in enumerate(filtered_files):
    
    if filename.endswith('.nc'):
        sim_time = (start + timedelta(minutes=15 * i)).strftime("%H-%M")
        file_path = os.path.join(sim_path, filename)
        #timestamp = filename.split('.')[-2]  # extrait 024 de ...OUT.024.nc par ex

        data = Dataset(file_path, 'r', format='NETCDF4')

        ALT = data.variables['level'][:] * 0.001  # km
        THT = data.variables['THT'][0, 1:-1, indicj, indici]
        theta = units.Quantity(THT, 'kelvin')  # <- Ajout ici
        
        PRE = data.variables['PABST'][0, 1:-1, indicj, indici]
        MRV = data.variables['RT'][0, 0, 1:-1, indicj, indici] * 1000.
        u = data.variables['UT'][0, 1:-1, indicj, indici]
        v = data.variables['VT'][0, 1:-1, indicj, indici]

        u = units.Quantity(u, "m/s")
        v = units.Quantity(v, "m/s")

        XRD = 287.05967
        XCPD = 1004.708845
        zp0 = 1000.

        z = units.Quantity(ALT, "meters")
        p = units.Quantity(PRE * 0.01, "hPa")
        T = units.Quantity(THT * (p / zp0) ** (XRD / XCPD) - 273.15, "degC")
        MRV = MRV * 0.001
        EV = MRV * p / (XRD / 461.5 + MRV)
        EV = np.where(EV <= 0, np.nan, EV)
        Td = units.Quantity(243.5 / (17.67 / np.log(EV / 6.112) - 1.), "degC")
        # Calcul du thermomètre mouillé
        Tw = mpcalc.wet_bulb_temperature(p, T, Td)
        theta_w = mpcalc.wet_bulb_potential_temperature(p, T, Td)



        fig = plt.figure(figsize=(9, 10))
        skew = SkewT(fig, rotation=45)

        skew.plot(p, T, 'r', label = '$T$')
        skew.plot(p, Td, 'cyan', label = '$T_d$')
        skew.plot(p, Tw, color='blue', linestyle='-', linewidth=2, label='$T_w$')        
        
        skew.ax.set_ylim(1000, 100)
        skew.ax.set_xlim(-40, 60)

        # Tracer isothermes manuelles
        temps_t = np.arange(-110, 70, 10)
        pres = np.linspace(1000, 100, 50)
        for t in temps_t:
            skew.ax.plot([t] * len(pres), pres, color='#e1ad01', linestyle='solid', linewidth=1)

        skew.ax.set_xlabel(f'Température ({T.units:~P})', fontsize=18)
        skew.ax.set_ylabel(f'Pression ({p.units:~P})', fontsize=18)

        lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
        skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black', label='LCL')

        print(f"Pression du LCL : {lcl_pressure.to('hPa').magnitude:.2f} hPa")


        prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
        skew.plot(p, prof, 'k', linewidth=2)

        cape, cin = mpcalc.cape_cin(p, T, Td, prof)
        
        
        cape_val = cape.to('J/kg').magnitude if cape is not None else np.nan
        cin_val = cin.to('J/kg').magnitude if cin is not None else np.nan
        cape_values.append(cape_val)
        cin_values.append(cin_val)

        # Store time label (adapt if you have real datetime info)
        timesteps.append(sim_time)
        i += 1

        skew.shade_cin(p, T, prof, Td)
        skew.shade_cape(p, T, prof)
        skew.ax.axvline(0, color='#e1ad01', linestyle='-', linewidth=2)

        # Adiabatiques
        skew.plot_dry_adiabats(linewidth=1.5, colors='#006400', linestyles='solid')
        skew.plot_moist_adiabats(linewidth=1.5, colors='#006400', linestyles='--')
        skew.plot_mixing_lines(linewidth=1.5, colors='#e1ad01', linestyles='--')  # lignes mélange = optionnelles

        fig.suptitle(f"Emagramme à {sim_time} au dessus du CRA: \n Longitude : 0° 22' 08'' E et Latitude : 43° 07' 45'' N" , fontsize=16, ha="center")
        skew.ax.text(0.02, 0.98,
                     f"CAPE : {cape.to('J/kg').magnitude:.1f} J kg$^{{-1}}$\nCIN : {cin.to('J/kg').magnitude:.1f} J kg$^{{-1}}$",
                     transform=skew.ax.transAxes,
                     fontsize=16,
                     verticalalignment='top',
                     horizontalalignment='left',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        skew.ax.legend(loc='upper right', fontsize=14, frameon=True)

        # Sauvegarde
        yfig = f"Emagramme_1/Emagramme_{sim_time}.png"
        fig.savefig(yfig, dpi=120)
        plt.close(fig)
        print(f"{yfig} sauvegardé.")

        with open(vertical_profile_csv, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for lvl in range(len(p)):
                writer.writerow([
                    sim_time,
                    lvl,
                    round(p[lvl].magnitude, 2),
                    round(T[lvl].magnitude, 2),
                    round(theta[lvl].magnitude, 2),
                    round(theta_w[lvl].magnitude, 2)
                ])



# === Sauvegarde CSV des valeurs CAPE/CIN ===
csv_file = os.path.join(output_folder, 'cape_cin_evolution.csv')
with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Heure', 'CAPE', 'CIN'])
    for t, c, ci in zip(timesteps, cape_values, cin_values):
        writer.writerow([t, c, ci])
print(f"{csv_file} sauvegardé.")

# === Tracer évolution temporelle CAPE et CIN ===
plt.figure(figsize=(10, 6))
plt.plot(timesteps, cape_values, marker='o', label='CAPE', color='red')
plt.plot(timesteps, cin_values, marker='x', label='CIN', color='blue')
plt.xticks(rotation=45, fontsize=8)
plt.ylabel('Énergie (J/kg)')
plt.title("Évolution temporelle de CAPE et CIN")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'evolution_CAPE_CIN.png'), dpi=120)
plt.close()
print("Graphique d'évolution CAPE/CIN sauvegardé.")
