# 1.read data from h5 files and save it to .npy files
#python read_data.py

# # 2.ptychographic reconstruction of 512x512 images
# for k in {0..174..4}; 
# do
#     for j in {0..3}; 
#     do
#         echo $(($k+$j))
#         CUDA_VISIBLE_DEVICES=$j python rec_ptycho_full.py $(($k+$j)) &                
#     done
#     wait
# done

# # 3. Sort projections with respect to angles
# python sort.py

# 4. Take data from rec_full_sorted and align it with sift in imageJ, save the result to rec_full_sorted_aligned/r_0000.tif

# 5. Find shifts between rec_full_sorted and rec_full_sorted_aligned, gives file shifts.npy
#python find_shifts_full.py

# 6. Recosruct with shifted scan positions and cropped
for k in {0..165..4}; 
do
    for j in {0..3}; 
    do
        echo $(($k+$j))
        CUDA_VISIBLE_DEVICES=$j python rec_ptycho_crop.py $(($k+$j)) &                
    done
    wait
done

# 7. Find shifts by using the com and sum overlines, gives file shiftssum.npy, check rec_crop_sum_check result
# python find_shifts_sum.py

# 8. find rotation center by using rec_crop_sum_check - 192
# python find_center.py

# 9. reconstruct all data with applying both shifts to scanning positions
# for k in {0..174..8}; 
# do
#     for j in {0..3}; 
#     do
#         echo $(($k+$j))
#         CUDA_VISIBLE_DEVICES=$j python rec_ptycho_crop_sum.py $(($k+$j)) &                
#     done
#     wait
# done


# 10. reconstruct with optical flow alignment
# python recall.py 200 sift
# python recall.py 200 stxm

# python recall.py 500 sift
# python recall.py 500 stxm

# python recall.py 1000 sift
# python recall.py 1000 stxm

#python recall.py 10 sift
#python recall.py 10 stxm

# python recall.py 6000 sift
# python recall.py 6000 stxm

# python recall.py 13689 sift
# python recall.py 13689 stxm

# python admm.py 200 sift
# # python admm.py 500 sift
# python admm.py 1000 sift
# # python admm.py 2000 sift
# # python admm.py 3800 sift

# python admm.py 200 sift
# # python admm.py 500 sift
# python admm.py 1000 sift