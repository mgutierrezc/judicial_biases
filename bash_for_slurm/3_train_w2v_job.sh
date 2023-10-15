#!/bin/sh
#
# Tests activation of virtual environment within a bash script

#
#SBATCH --account=sscc         # Replace ACCOUNT with your group account name
#SBATCH --job-name=train_models     # The job name.
#SBATCH -c 4                      # The number of cpu cores to use
#SBATCH -t 0-02:00                 # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=16gb         # The memory the job will use per cpu core

source ~/.bashrc
module load anaconda
conda activate glove_env

echo "-- running model trainer --"
python /burg/home/mg4558/others/tests/repos/peru_cases_iat/3_train_embeddings_w2v.py
echo "-- model trainer completed run --"