# Collection of scripts to calculate spectral fluxes averaged over every selected IASI orbit. 
It currently calculates the mean spectral fluxes over eight domains, namely
all combinations of:
    1) global and tropics
    2) all-sky and clear-sky
    3) land+ocean and ocean-only
The script takes three input parameters:
    1) the year,
    2) the month,
    3) and the day

The main processing is conducted in process_iasi.py. It calls the read routine in read_iasi.py to read the IASI data. It then filters the data (see above), averages the spectral radiances and calculates spectral fluxes using angular interpolation and integration. It saves the mean flux and the number of pixels it is based on to .npy (numpy) type files. The script postprocess_iasi.py uses those files to calculate monthly and yearly averages which are needed to calculate the spectral feedback parameter. The script start_jobs.sh is an example for calling process_iasi.py using a workload manager (here the SLURM manager used on DRKZ's mistral server). This needs to be adapted if run in a different environment.
