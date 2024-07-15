#!/bin/bash

##############################################
## BEFORE SUBMITTING THIS SCRIPT: do
##    $ srun --pty /bin/bash
##    $ newgrp docker
##    $ conda activate alphafold
#     $ docker run --rm --gpus all nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04 nvidia-smi   # use this to get GPU device id, should go in order to have slurm allocates (likely) the same GPUs as one actually using.
##    $ sbatch alphafold.sh
##############################################

# INFOS

#SBATCH --export=ALL                         
#SBATCH --uid=tlp5359

#SBATCH -J alphafold                               	               # Job name
#SBATCH -o _slurm_output/alphafold.out-%J                          # stdout file name
#SBATCH -e _slurm_output/alphafold.out-%J                          # stderr file name

### Node info
#SBATCH --partition=normal                    # Queue name (normal, conference)
#SBATCH --nodes=1                                                            
#SBATCH --ntasks-per-node=1                   # Number of tasks per node
#SBATCH -t 7-0:00:00                          # Run time (d-hh:mm:ss)


###############

# RESOURCES

#SBATCH --mem=30G                            # Memory requirements
#SBATCH --cpus-per-task=24                   # Number of cpus needed per task
#SBATCH --gres=gpu:a100:1                    # Number of gpus needed


ALPHAFOLD_DIR="/raid/AlphaFold/alphafold"
FASTA_DIR="/home/tlp5359/projects/CC_Sequencing_LLM/BCB/pdbs/predicted_sequences"
OUT_DIR="/home/tlp5359/projects/CC_Sequencing_LLM/BCB/pdbs/new_ones/"
mkdir -p $OUT_DIR

DOWNLOAD_DIR="/raid/AlphaFold/databases"
cd $ALPHAFOLD_DIR


# input_lst="${FASTA_DIR}/UniRef100_A0A6C9QGD2.fasta,${FASTA_DIR}/UniRef100_A0A377CN25.fasta,${FASTA_DIR}/UniRef100_A0A1Q6APE6.fasta,${FASTA_DIR}/UniRef100_A0A376U1T7.fasta,${FASTA_DIR}/UniRef100_A0A140JY95.fasta,${FASTA_DIR}/UniRef100_V0VTN2.fasta,${FASTA_DIR}/UniRef100_A0A641J846.fasta,${FASTA_DIR}/UniRef100_A0A376J6F5.fasta"
# input_lst="${FASTA_DIR}/UniRef100_A0A826TFI3.fasta,${FASTA_DIR}/UniRef100_A0A376RNM9.fasta,${FASTA_DIR}/UniRef100_A0A854BY61.fasta,${FASTA_DIR}/UniRef100_A0A376VH98.fasta,${FASTA_DIR}/UniRef100_A0A8B5RFA4.fasta,${FASTA_DIR}/UniRef100_A0A1M2DQG9.fasta,${FASTA_DIR}/UniRef100_V0A0E6.fasta,${FASTA_DIR}/UniRef100_A0A826SGH6.fasta,${FASTA_DIR}/UniRef100_A0A789MCR5.fasta"
input_lst="${FASTA_DIR}/UniRef100_A0A0K5GZQ2.fasta,${FASTA_DIR}/UniRef100_A0A2Z2JDV5.fasta,${FASTA_DIR}/UniRef100_A0A0K9T562.fasta,${FASTA_DIR}/UniRef100_A0A826R8K6.fasta,${FASTA_DIR}/UniRef100_J7QWZ8.fasta,${FASTA_DIR}/UniRef100_A0A827L924.fasta,${FASTA_DIR}/UniRef100_A0A2X7G8D6.fasta,${FASTA_DIR}/UniRef100_A0A788Y284.fasta,${FASTA_DIR}/UniRef100_A0A1M0D635.fasta,${FASTA_DIR}/UniRef100_A0A0J1ZBG5.fasta,${FASTA_DIR}/UniRef100_A0A193LM41.fasta,${FASTA_DIR}/UniRef100_A0A376TK98.fasta,${FASTA_DIR}/UniRef100_C0M637.fasta"

python3 docker/run_docker.py \
  --fasta_paths=$input_lst \
  --db_preset=reduced_dbs \
  --max_template_date=2022-01-01 \
  --model_preset=monomer \
  --data_dir=$DOWNLOAD_DIR \
  --output_dir=$OUT_DIR  --gpu_devices=5


# python3 docker/run_docker.py \
#   --fasta_paths="${FASTA_DIR}/UniRef100_A0A5F1DTN5.fasta,${FASTA_DIR}/UniRef100_B7M3T9.fasta"  \
#   --db_preset=reduced_dbs \
#   --max_template_date=2022-01-01 \
#   --model_preset=monomer \
#   --data_dir=$DOWNLOAD_DIR \
#   --output_dir=$OUT_DIR
