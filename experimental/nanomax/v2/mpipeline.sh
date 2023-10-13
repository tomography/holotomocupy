# python read_data.py
# python sort.py
# for k in {0..174..32}; 
# do
#     for j in {0..31}; 
#     do
#         echo $(($k+$j))
#         python pgm1.py $(($k+$j)) &                
#     done
#     wait
# done
# python prealign_sift1.py

# for k in {0..174..32}; 
# do
#     for j in {0..31}; 
#     do
#         echo $(($k+$j))
#         python pgm2.py $(($k+$j)) &                
#     done
#     wait
# done
# python prealign_sift2.py

# for k in {0..174..32}; 
# do
#     for j in {0..31}; 
#     do
#         echo $(($k+$j))
#         python pgm3.py $(($k+$j)) &                
#     done
#     wait
# done
for k in {8..174..8}; 
do
    for j in {0..7}; 
    do
        echo $(($k+$j))
        CUDA_VISIBLE_DEVICES=$j python rec_full.py $(($k+$j)) &                
    done
    wait
done

# python prealign_crop_sift.py

# for k in {0..174..16}; 
# do
#     for j in {0..16}; 
#     do
#         echo $(($k+$j))
#         python pgm2_crop.py $(($k+$j)) &                
#     done
#     wait
# done

# python prealign_crop_cm.py


# for k in {88..174..4}; 
# do
#     for j in {0..3}; 
#     do
#         echo $(($k+$j))
#         CUDA_VISIBLE_DEVICES=$j python rec_crop.py $(($k+$j)) &                
#     done
#     wait
# done

# 6. Find shifts between reccrop and reccrop_sift, gives file shiftscrop.npy
# python find_shifts_crop.py

# -. Find shifts by using the com and sum overlines, gives file shiftssum.npy, check rec_crop_sum_check result
# python find_shifts_sum.py

# 8. find rotation center by using rec_crop_sum_check - 190
# python find_center.py

# 9. reconstruct all data with applying both shifts to scanning positions
# for k in {0..174..4}; 
# do
#     for j in {0..3}; 
#     do
#         echo $(($k+$j))
#         CUDA_VISIBLE_DEVICES=$j python reccrop_align_azat.py $(($k+$j)) &                
#     done
#     wait
# done
# python find_center.py


# 10. reconstruct with optical flow alignment
# python admm.py 500 sift 190 1
# python admm.py 500 sift 190 0

# python admm.py 3800 sift 190 1
# python admm.py 3800 sift 190 0
# python admm.py 200 stxm 190 1

# python admm.py 500 sift 190
# python admm.py 500 stxm

# python admm.py 1000 sift
# python admm.py 1000 stxm

#python admm.py 10 sift
#python admm.py 10 stxm

# python admm.py 6000 sift
# python admm.py 6000 stxm

# python admm.py 13689 sift
# python admm.py 13689 stxm

# python admm.py 200 sift
# # python admm.py 500 sift
# python admm.py 1000 sift
# # python admm.py 2000 sift
# # python admm.py 3800 sift

# python admm.py 200 sift
# # python admm.py 500 sift
# python admm.py 1000 sift