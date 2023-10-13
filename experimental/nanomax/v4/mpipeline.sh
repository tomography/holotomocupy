#!/bin/bash
source /home/vvnikitin/anaconda3/etc/profile.d/conda.sh
conda activate tomoalign
prefix=/projects/nanoct/nanomax/ 
#python read_data.py $prefix
#python sort.py $prefix
#for k in {0..174..8}; 
#do
#    for j in {0..7}; 
#    do
#        echo $(($k+$j))
#        CUDA_VISIBLE_DEVICES=$j python rec_full.py $prefix $(($k+$j)) 2000 1 16 &                
#    done
#    wait
#done
#python prealign_sift1.py $prefix 2000


for k in {0..174..8}; 
do
    for j in {0..7}; 
    do
        echo $(($k+$j))
        CUDA_VISIBLE_DEVICES=$j python rec_crop.py $prefix $(($k+$j)) 2700 1 64 &                
    done
    wait
done
python prealign_sift2.py $prefix 2700

for k in {0..174..8}; 
do
    for j in {0..7}; 
    do
        echo $(($k+$j))
        CUDA_VISIBLE_DEVICES=$j python rec_crop_final.py $prefix  $(($k+$j)) 4000 1 256 &                
    done
    wait
done

