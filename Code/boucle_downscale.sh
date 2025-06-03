cd /Documents/STAGE/

#module load python/2.7.3 


# Grille MSG3
grille_file=../Observation/MSG+00003km.nc

# Code python  à executer
code=/home/etudiant/Documents/STAGE/Plot/downscale_1.py 

# Répertoires 
sim_path="../Simulations/Sim_2_CRA"
obs_path="../Observation/Sim_2_CRA_nc"

# Debut des indices
start_sim=1
start_obs=53
pairs=37 		#Nombre des couples d'observation


mkdir -p DOWN_OUTPUT

for ((j=0; j<pairs; j++)); do
	obs_index=$(printf "%03d" $((start_obs + j)))
	obs_file="$obs_path/Mmultic3kmNC4_msg03_${obs_index}.nc"

	# Deux fichiers simulation à associer 
	sim1_index=$(printf "%03d" $((start_sim + j*2)))
	sim2_index=$(printf "%03d" $((start_sim + j*2 + 1)))

	sim_file1="$sim_path/FLY15.1.SEP13.OUT.${sim1_index}.nc"
	sim_file2="$sim_path/FLY15.1.SEP13.OUT.${sim2_index}.nc"
	
	# Traitement des deux fichiers SIM avec le même fihcier OBS

	for sim_file in "$sim_file1" "$sim_file2"; do
		python3 "$code" "$sim_file" "$obs_file" "$grille_file" -s -o DOWN_OUTPUT/
	done


done 
