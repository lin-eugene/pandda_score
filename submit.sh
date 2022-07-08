#!/bin/bash

export devpath=/dls/labxchem/data/2018/lb18145-80/processing/analysis/eugene/namdinator_test
export SINGULARITY_BIND="/dls/labxchem/data/2018/lb18145-80/processing/analysis/eugene/namdinator_test:/mnt"
cd $devpath
singularity exec --nv namdinator.sif /bin/sh namdinator_run.sh