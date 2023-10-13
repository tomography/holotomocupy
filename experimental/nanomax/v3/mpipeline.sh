python read_data.py
python sort.py
# for k in {0..174..8}; 
# do
#     for j in {0..7}; 
#     do
#         echo $(($k+$j))
#         CUDA_VISIBLE_DEVICES=$j python rec_full.py $(($k+$j)) 2700 1 16 &                
#     done
#     wait
# done
# python prealign_sift1.py 2700


# for k in {0..174..8}; 
# do
#     for j in {0..7}; 
#     do
#         echo $(($k+$j))
#         CUDA_VISIBLE_DEVICES=$j python rec_crop.py $(($k+$j)) 2700 1 64 &                
#     done
#     wait
# done
# python prealign_sift2.py 2700
for k in {0..174..8}; 
do
    for j in {0..7}; 
    do
        echo $(($k+$j))
        CUDA_VISIBLE_DEVICES=$j python rec_crop2.py $(($k+$j)) 4000 1 32 &                
    done
    wait
done

python find_center.py 2700
