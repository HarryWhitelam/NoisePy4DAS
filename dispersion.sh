#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gfs19eku@uea.ac.uk
#SBATCH -p compute-64-512
#SBATCH --mem=20G
#SBATCH --time=4-00:00
#SBATCH -o results/dispersion-%j.out
#SBATCH -e results/dispersion-%j.err

# module load python/anaconda/2023.07/3.11.4
# conda activate SeaDASxCorr
python scripts/dispersion.py
