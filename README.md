# Collection of scripts to calculate spectral fluxes averaged over every selected IASI orbit. It allows calculates the spectral fluxes for three different domains:
    1) all-sky in the whole tropics ("tropics")
    2) all-sky over the tropical ocean ("ocean")
    3) clear-sky over the tropical ocean ("clear")

The main processing is conducted in process_iasi.py. It calls the read routine in read_iasi.py to read the IASI data. It then filters the data (see above), averages the spectral radiances and calculates spectral fluxes using angular interpolation and integration. It saves the mean flux and the number of pixels it is based on to .npy (numpy) type files. The script postprocess_iasi.py uses those files to calculate monthly and yearly averages which are needed to calculate the spectral feedback parameter. The script start_jobs.sh is an example for calling process_iasi.py using a workload manager (here the SLURM manager used on DRKZ's mistral server). This needs to be adapted if run in a different environment.
