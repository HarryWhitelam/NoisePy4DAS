#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gfs19eku@uea.ac.uk
#SBATCH -p compute-64-512
#SBATCH -n 8
#SBATCH --mem=48G
#SBATCH --time=7-00:00
#SBATCH -o results/correlation-%j.out
#SBATCH -e results/correlation-%j.err

# module load python/anaconda/2023.07/3.11.4
# conda activate seismic
python scripts/correlation.py
