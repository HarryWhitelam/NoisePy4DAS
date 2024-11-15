#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gfs19eku@uea.ac.uk
#SBATCH -p compute-64-512
#SBATCH --mem=32G
#SBATCH --time=4-00:00
#SBATCH -o results/visualisation-%j.out
#SBATCH -e results/visualisation-%j.err

# module load python/anaconda/2023.07/3.11.4
# conda activate SeaDASxCorr
python scripts/visualisation.py
