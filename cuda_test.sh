#!/bin/bash
#PBS -q default
#PBS -P PanddaScore
#PBS -j oe
#PBS -N Job_Name_1
#PBS -l select=1:ncpus=2:mem=10gb:ngpus=1
#PBS -l walltime=24:00:00

np=$(cat ${PBS_NODEFILE} | wc -l);

conda activate pandda_env
cd /dls/labxchem/data/2018/lb18145-80/processing/analysis/eugene/panddaproject
python cuda_test.py