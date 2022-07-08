#!/bin/bash

mkdir /exports/vzu76495/tmp
mkdir /exports/vzu76495/tmpfs
mkdir /exports/vzu76495/cache

export SINGULARITY_TMPDIR=/exports/vzu76495/tmp
export SINGULARITY_CACHEDIR=/exports/vzu76495/cache
export SINGULARITY_BIND="/exports/vzu76495/tmpfs:/tmp"
cd /dls/labxchem/data/2018/lb18145-80/processing/analysis/eugene/namdinator_test/

singularity build -f -F /dls/labxchem/data/2018/lb18145-80/processing/analysis/eugene/namdinator_test/namdinator.sif /dls/labxchem/data/2018/lb18145-80/processing/analysis/eugene/namdinator_test/namdinator.def
