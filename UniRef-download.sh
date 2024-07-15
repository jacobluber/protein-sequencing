#!/bin/bash
  
# INFOS

#SBATCH --export=ALL                         # Start with user's environment
#SBATCH --get-user-env
#SBATCH --uid=tlp5359

#SBATCH -J UNIREF-download                       # Job name
#SBATCH -o downloadUniRef.out-%j                 # stdout file name
#SBATCH -e downloadUniRef.out-%j                 # stderr file name

#SBATCH --mail-user=tlp5359@mavs.uta.edu      # Email of notification
#SBATCH --mail-type=END,FAIL,REQUEUE                                      

### Node info
#SBATCH --partition=normal                    # Queue name (normal, conference)
#SBATCH --nodes=1                                                            
#SBATCH --ntasks-per-node=1                   # Number of tasks per node
#SBATCH -t 7-0:00:00                          # Run time (d-hh:mm:ss)


###############

# RESOURCES

#SBATCH --gres=gpu:0                          # Number of gpus needed
#SBATCH --mem=15G                             # Memory requirements
#SBATCH --cpus-per-task=24                     # Number of cpus needed per task


################
## START OF EXECUTIONS

WKDIR="/home/tlp5359/data-UniRef/"
mkdir -p $WKDIR
cd $WKDIR

# ---
# Define an associative array for taxon_id and corresponding names
declare -A taxon_map
taxon_map[208962]="E-albertii"
taxon_map[564]="E-fergusonii"

# identity levels for UniRef100, UniRef90, and UniRef50
identities=(1.0)   


# Loop through all items in the associative array
for taxon_id in "${!taxon_map[@]}"; do
  name=${taxon_map[$taxon_id]}

  for identity in "${identities[@]}"; do
    rounded_i=$(awk -v id="$identity" 'BEGIN { printf("%.0f", id * 100); }')

    out_name=${name}_${taxon_id}_UniRef${rounded_i}.fasta
    echo "WORKING ON $out_name"

    # Set the query parameters
    query=%28%28taxonomy_id%3A${taxon_id}%29+AND+%28identity%3A${identity}%29+NOT+%28name%3A%22hypothetical+protein%22%29%29
    echo "curl \"https://rest.uniprot.org/uniref/stream?format=fasta&query=${query}\""
    
    #curl on background (delete the & at the end if don't want to)
    curl "https://rest.uniprot.org/uniref/stream?format=fasta&query=${query}" > $out_name &
  done

done
# Wait for all background jobs to complete
wait

# # make it public 
# chmod -R 777 *
# echo "All .fasta files have been made publicly readable."
