
arr=(1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)
for k in $(seq 0 2 34); do 
    CUDA_VISIBLE_DEVICES=0 python code_tests.py 2048 ${arr[$k]} 3 3 True True 1e-10  &
    CUDA_VISIBLE_DEVICES=1 python code_tests.py 2048 ${arr[$(($k+1))]} 3 3 True True 1e-10 &
    wait
done

for k in $(seq 0 2 34); do 
    CUDA_VISIBLE_DEVICES=0 python code_tests.py 2048 ${arr[$k]} 4 3 True True 1e-10  &
    CUDA_VISIBLE_DEVICES=1 python code_tests.py 2048 ${arr[$(($k+1))]} 4 3 True True 1e-10 &
    wait
done

for k in $(seq 0 2 34); do 
    CUDA_VISIBLE_DEVICES=0 python code_tests.py 2048 ${arr[$k]} 3 3 False True 1e-10  &
    CUDA_VISIBLE_DEVICES=1 python code_tests.py 2048 ${arr[$(($k+1))]} 3 3 False True 1e-10 &
    wait
done

for k in $(seq 0 2 34); do 
    CUDA_VISIBLE_DEVICES=0 python code_tests.py 2048 ${arr[$k]} 4 3 False True 1e-10  &
    CUDA_VISIBLE_DEVICES=1 python code_tests.py 2048 ${arr[$(($k+1))]} 4 3 False True 1e-10 &
    wait
done