#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gfs19eku@uea.ac.uk
#SBATCH -p compute-24-96
#SBATCH --mem=24G
#SBATCH --time=3-00:00
#SBATCH -o results/correlation-%j.out
#SBATCH -e results/correlation-%j.err

# conda activate NoisePy4DAS-SeaDAS
python scripts/correlation.py
