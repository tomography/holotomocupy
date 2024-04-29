python data_modeling_chip_codes.py 17.05 4e-3 1e-6 True
python data_modeling_chip_codes.py 33.35 4e-3 1e-6 True

python data_modeling_chip_codes.py 17.05 4e-3 1e-6 False
python data_modeling_chip_codes.py 33.35 4e-3 1e-6 False

python data_modeling_chip_codes.py 17.05 8e-3 1e-6 True
python data_modeling_chip_codes.py 33.35 8e-3 1e-6 True

python data_modeling_chip_codes.py 17.05 8e-3 1e-6 False
python data_modeling_chip_codes.py 33.35 8e-3 1e-6 False

python data_modeling_chip_codes.py 17.05 270e-3 5e-5 True
python data_modeling_chip_codes.py 33.35 270e-3 5e-5 True

python data_modeling_chip_codes.py 17.05 270e-3 5e-5 False
python data_modeling_chip_codes.py 33.35 270e-3 5e-5 False


#tomo4
CUDA_VISIBLE_DEVICES=0 nohup python iterative_reconstruction_chip_codes_admm.py 17.05 4e-3 1e-6 True 8192 64 >log/r000 2>log/r000 &
CUDA_VISIBLE_DEVICES=1 nohup python iterative_reconstruction_chip_codes_admm.py 33.35 4e-3 1e-6 True 8192 64 >log/r001 2>log/r001 &
CUDA_VISIBLE_DEVICES=2 nohup python iterative_reconstruction_chip_codes_admm.py 17.05 4e-3 1e-6 False 8192 64 >log/r002 2>log/r002 &
CUDA_VISIBLE_DEVICES=3 nohup python iterative_reconstruction_chip_codes_admm.py 33.35 4e-3 1e-6 False 8192 64 >log/r003 2>log/r003 &

#tomo5
CUDA_VISIBLE_DEVICES=0 nohup python iterative_reconstruction_chip_codes_admm.py 17.05 8e-3 1e-6 True 8192 64 >log/r004 2>log/r004 &
CUDA_VISIBLE_DEVICES=1 nohup python iterative_reconstruction_chip_codes_admm.py 33.35 8e-3 1e-6 True 8192 64 >log/r005 2>log/r005 &
CUDA_VISIBLE_DEVICES=2 nohup python iterative_reconstruction_chip_codes_admm.py 17.05 8e-3 1e-6 False 8192 64 >log/r006 2>log/r006 &
CUDA_VISIBLE_DEVICES=3 nohup python iterative_reconstruction_chip_codes_admm.py 33.35 8e-3 1e-6 False 8192 64 >log/r007 2>log/r007 &

#tomo2
CUDA_VISIBLE_DEVICES=0 nohup python iterative_reconstruction_chip_codes_admm.py 17.05 270e-3 5e-5 True 8192 64 >log/r008 2>log/r008 &
CUDA_VISIBLE_DEVICES=1 nohup python iterative_reconstruction_chip_codes_admm.py 33.35 270e-3 5e-5 True 8192 64 >log/r009 2>log/r009 &

#tomo1
CUDA_VISIBLE_DEVICES=0 nohup python iterative_reconstruction_chip_codes_admm.py 17.05 270e-3 5e-5 False 8192 64 >log/r010 2>log/r010 &

#tomo3
CUDA_VISIBLE_DEVICES=0 nohup python iterative_reconstruction_chip_codes_admm.py 33.35 270e-3 5e-5 False 8192 64 >log/r011 2>log/r011 &



energy = float(sys.argv[1])  # [keV] xray energy
z1p = float(sys.argv[2])# positions of the probe and code for reconstruction
# z1p = 270e-3# positions of the probe and code for reconstruction
ill_feature_size = float(sys.argv[3])
use_prb = sys.argv[4]=='True'
use_code = sys.argv[5]=='True'
ndist = int(sys.argv[6])

#tomo3

python data_modeling_chip_codes.py 17.05 4e-3 1e-6 True False 3
python data_modeling_chip_codes.py 33.35 4e-3 1e-6 True False 3
python data_modeling_chip_codes.py 17.05 4e-3 1e-6 False False 3
python data_modeling_chip_codes.py 33.35 4e-3 1e-6 False False 3

CUDA_VISIBLE_DEVICES=0 nohup python iterative_reconstruction_chip_codes_admm.py 17.05 4e-3 1e-6 True False 2 1024 64 >log/r016 2>log/r016 &
CUDA_VISIBLE_DEVICES=1 nohup python iterative_reconstruction_chip_codes_admm.py 33.35 4e-3 1e-6 True False 2 1024 64 >log/r017 2>log/r017 &
CUDA_VISIBLE_DEVICES=2 nohup python iterative_reconstruction_chip_codes_admm.py 17.05 4e-3 1e-6 False False 2 1024 64 >log/r018 2>log/r018 &
CUDA_VISIBLE_DEVICES=3 nohup python iterative_reconstruction_chip_codes_admm.py 33.35 4e-3 1e-6 False False 2 1024 64 >log/r019 2>log/r019 &

#tomo2
CUDA_VISIBLE_DEVICES=0 nohup python iterative_reconstruction_chip_codes_admm.py 17.05 4e-3 1e-6 True False 3 1024 64 >log/r012 2>log/r012 &
CUDA_VISIBLE_DEVICES=1 nohup python iterative_reconstruction_chip_codes_admm.py 33.35 4e-3 1e-6 True False 3 1024 64 >log/r013 2>log/r013 &

#tomo1
CUDA_VISIBLE_DEVICES=0 nohup python iterative_reconstruction_chip_codes_admm.py 17.05 4e-3 1e-6 False False 3 1024 64 >log/r014 2>log/r014 &
CUDA_VISIBLE_DEVICES=1 nohup python iterative_reconstruction_chip_codes_admm.py 33.35 4e-3 1e-6 False False 3 1024 64 >log/r015 2>log/r015 &



