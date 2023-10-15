#!/bin/sh
#
# Tests activation of virtual environment within a bash script

#
#SBATCH --account=sscc         # Replace ACCOUNT with your group account name
#SBATCH --job-name=create_data     # The job name.
#SBATCH -c 4                      # The number of cpu cores to use
#SBATCH -t 1-10:00                 # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=30gb         # The memory the job will use per cpu core

source ~/.bashrc
module load anaconda
conda activate glove_env

echo "-- running data cleaner --"
python /burg/home/mg4558/others/tests/repos/peru_cases_iat/1_create_judges_files.py
echo "-- data cleaner completed run --"