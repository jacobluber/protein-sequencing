#!/bin/bash

# INFOS

#SBATCH --export=ALL                         # Start with user's environment
#SBATCH --get-user-env
#SBATCH --uid=tlp5359

#SBATCH -J altrain-EC-M                                # Job name
#SBATCH -o _slurm_output/train.out-%J                          # stdout file name
#SBATCH -e _slurm_output/train.out-%J                          # stderr file name

#SBATCH --mail-user=tlp5359@mavs.uta.edu      # Email of notification
#SBATCH --mail-type=END,FAIL,REQUEUE                                      

### Node info
#SBATCH --partition=normal                    # Queue name (normal, conference)
#SBATCH --nodes=1                                                            
#SBATCH --ntasks-per-node=1                   # Number of tasks per node
#SBATCH -t 7-0:00:00                          # Run time (d-hh:mm:ss)


###############

# RESOURCES

#SBATCH --gres=gpu:a100:1                    # Number of gpus needed
#SBATCH --mem=30G                            # Memory requirements
#SBATCH --cpus-per-task=16                   # Number of cpus needed per task


################

## START OF EXECUTIONS


DATA_DIR=
PROJECT_DIR="/home/tlp5359/projects/CC_Sequencing_LLM/BCB/"
OUTPUT_DIR=

# run da script
# cat $PROJECT_DIR/bcb-training.py
# time python3 $PROJECT_DIR/bcb-training.py
cat $PROJECT_DIR/bcb-training-albert.py
time python3 $PROJECT_DIR/bcb-training-albert.py