#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gfs19eku@uea.ac.uk
#SBATCH -p compute-64-512
#SBATCH --mem=48G
#SBATCH --time=4-00:00
#SBATCH -o results/phase_shift-%j.out
#SBATCH -e results/phase_shift-%j.err

# module load python/anaconda/2023.07/3.11.4
# conda activate SeaDASxCorr
python scripts/phase_shift.py
