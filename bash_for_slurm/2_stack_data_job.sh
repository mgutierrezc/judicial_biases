#!/bin/sh
#
# Tests activation of virtual environment within a bash script

#
#SBATCH --account=sscc         # Replace ACCOUNT with your group account name
#SBATCH --job-name=stacker_data     # The job name.
#SBATCH -c 4                      # The number of cpu cores to use
#SBATCH -t 1-00:00                 # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=16gb         # The memory the job will use per cpu core

source ~/.bashrc
module load anaconda
conda activate glove_env

echo "-- running data stacker --"
python /burg/home/mg4558/others/tests/repos/peru_cases_iat/2_judges_files_stacker.py
echo "-- data stacker completed run --"