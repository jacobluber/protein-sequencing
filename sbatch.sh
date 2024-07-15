#!/bin/bash

#SBATCH --nodes=1                    # node count
#SBATCH --ntasks=1                   # total number of tasks across all nodes
#SBATCH --cpus-per-task=4            # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=25G
#SBATCH --output=_slurm_output/inference.out-%j
#SBATCH --gres=gpu:a100_1g.10gb:1              #a100_2g.20gb:1   #a100_1g.10gb:1 # a100_1g.10gb:2  #:a100:1
#SBATCH --time=7-0:00:00


# #SBATCH --job-name=test-KCYM              # create a short name for your job
# cat 2-TESTING-inference-KCYM.py
# time python3 2-TESTING-inference-KCYM.py


# #SBATCH --job-name=test-KCYMRHWST
# cat 2-TESTING-inference-KCYMRHWST.py
# time python3 2-TESTING-inference-KCYMRHWST.py



# #SBATCH --job-name=test-genH
# cat 2-TESTING-inference-for-gen-table.py
# time python3 2-TESTING-inference-for-gen-table.py


#SBATCH --job-name=test-albert
cat 2-TESTING-inference-Albert.py
time python3 2-TESTING-inference-Albert.py


#  clear; grep -A 1000 "PERFORM FOLD #0" _slurm_output/slurm_output-11796
