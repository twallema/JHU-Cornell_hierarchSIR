#!/bin/bash
#SBATCH --job-name=incremental-calibration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=72:00:00

# Submit as follows:
# sbatch --export=ALL,STRAINS=1,IMMUNITY_LINKING=False,USE_ED_VISITS=False submit_forecast.sh

# Pin the number of cores for use in python calibration script
export NUM_CORES=$SLURM_CPUS_PER_TASK

# Load any necessary modules
module load boost
module load anaconda3

# Activate the virtual environment
conda activate HIERARCHSIR

# Run your Python script
python incremental_forecasting.py --strains $STRAINS --immunity_linking $IMMUNITY_LINKING --use_ED_visits $USE_ED_VISITS

# Deactivate the virtual environment after the run
conda deactivate