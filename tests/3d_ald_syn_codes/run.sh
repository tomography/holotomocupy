# energy = float(sys.argv[1])  # [keV] xray energy
# z1p = float(sys.argv[2])# positions of the probe and code for reconstruction
# # z1p = 270e-3# positions of the probe and code for reconstruction
# ill_feature_size = float(sys.argv[3])
# use_prb = sys.argv[4]=='True'
# use_code = sys.argv[5]=='True'
# ndist = int(sys.argv[6])
# smooth = int(sys.argv[7])
# flg_show = False

python data_modeling.py 25 8e-3 1e-6 True True 1 0 
python data_modeling.py 25 8e-3 1e-6 True True 1 5 
python data_modeling.py 25 8e-3 1e-6 True True 1 10 
python data_modeling.py 25 8e-3 1e-6 True True 1 20 
python data_modeling.py 25 8e-3 1e-6 True True 1 50 
python data_modeling.py 25 8e-3 1e-6 True True 1 100 
python data_modeling.py 25 8e-3 1e-6 True True 1 300 


python data_modeling.py 25 8e-3 1e-6 False True 1 0 
python data_modeling.py 25 8e-3 1e-6 False True 1 5 
python data_modeling.py 25 8e-3 1e-6 False True 1 10 
python data_modeling.py 25 8e-3 1e-6 False True 1 20 
python data_modeling.py 25 8e-3 1e-6 False True 1 50 
python data_modeling.py 25 8e-3 1e-6 False True 1 100 
python data_modeling.py 25 8e-3 1e-6 False True 1 300 



energy = float(sys.argv[1])  # [keV] xray energy
z1p = float(sys.argv[2])# positions of the probe and code for reconstruction
# z1p = 270e-3# positions of the probe and code for reconstruction
ill_feature_size = float(sys.argv[3])
use_prb = sys.argv[4]=='True'
use_code = sys.argv[5]=='True'
ndist = int(sys.argv[6])
niter = int(sys.argv[7])
step = int(sys.argv[8])
smooth = int(sys.argv[9])
flg_show = False

tomo1
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 25 8e-3 1e-6 True True 1 100000 256 100 >True_100 2>True_100 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 25 8e-3 1e-6 False True 1 100000 256 100 >False_100 2>False_100 &

tomo4
CUDA_VISIBLE_DEVICES=0 nohup python rec_admm.py 25 8e-3 1e-6 False True 1 100000 256 0 >False_0 2>False_0 &
CUDA_VISIBLE_DEVICES=1 nohup python rec_admm.py 25 8e-3 1e-6 True True 1 100000 256 0 >True_0 2>True_0 &
CUDA_VISIBLE_DEVICES=2 nohup python rec_admm.py 25 8e-3 1e-6 False True 1 100000 256 10 >False_10 2>False_10 &
CUDA_VISIBLE_DEVICES=3 nohup python rec_admm.py 25 8e-3 1e-6 True True 1 100000 256 10 >True_10 2>True_10 &


