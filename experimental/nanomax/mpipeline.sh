# 1.read data from h5 files and save it to .npy files
python read_data.py

# 3. Sort projections with respect to angles
python sort.py

#4. Take data from rec_full_sorted and align it with sift in imageJ, save the result to recfull_sorted_sift/r_00000.tiff
#cd /data/staff/tomograms/vviknik/nanomax/recfull_sorted_sift
#for k in $(ls); do mv $k $(echo $k | sed 's/tif/tiff/g'); done

# 5. Find shifts between rec_full_sorted and rec_full_sorted_aligned, gives file shifts.npy
# python find_shifts_full.py

# 6. Recosruct with shifted scan positions and cropped
# for k in {0..165..8}; 
# do
#     for j in {0..3}; 
#     do
#         echo $(($k+$j))
#         CUDA_VISIBLE_DEVICES=$j python reccrop.py $(($k+$j)) &                
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
for k in {0..174..4}; 
do
    for j in {0..3}; 
    do
        echo $(($k+$j))
        CUDA_VISIBLE_DEVICES=$j python reccrop_align.py $(($k+$j)) &                
    done
    wait
done
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