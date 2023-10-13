#!/bin/bash
source /home/vvnikitin/anaconda3/etc/profile.d/conda.sh
conda activate tomoalign

prefix=/projects/nanoct/nanomax/

if [[ $1 -eq 1 ]]
then
    NUMEXPR_MAX_THREADS=128 python admm.py $prefix 4400 192 1 1
fi
if [[ $1 -eq 2 ]]
then
    NUMEXPR_MAX_THREADS=128 python admm.py $prefix 4400 192 0 1
fi
if [[ $1 -eq 3 ]]
then
NUMEXPR_MAX_THREADS=128 python admm.py $prefix 2200 192 1 2
fi
if [[ $1 -eq 4 ]]
then
NUMEXPR_MAX_THREADS=128 python admm.py $prefix 2200 192 0 2
fi
if [[ $1 -eq 5 ]]
then
NUMEXPR_MAX_THREADS=128 python admm.py $prefix 1100 192 1 4
fi
if [[ $1 -eq 6 ]]
then
NUMEXPR_MAX_THREADS=128 python admm.py $prefix 1100 192 0 4
fi
if [[ $1 -eq 7 ]]
then
NUMEXPR_MAX_THREADS=128 python admm.py $prefix 600 192 1 8
fi
if [[ $1 -eq 8 ]]
then
NUMEXPR_MAX_THREADS=128 python admm.py $prefix 600 192 0 8
fi
    
