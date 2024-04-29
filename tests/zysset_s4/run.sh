
# cp.cuda.Device(int(sys.argv[1])).use()
# ntheta = int(sys.argv[2])#750  # number of angles (rotations)
# ptheta = int(sys.argv[3])  # holography chunk size for GPU processing
# binning = int(sys.argv[4])
# niter = int(sys.argv[5])
# iter_step = int(sys.argv[6])
# same_probe = sys.argv[7]=='True'
# rand = sys.argv[8]
# shifts_probe_flg = sys.argv[9]=='True'
# st = int(sys.argv[10])

tomo4

nohup python rec.py 0 375 25 0 257 64 True norand True 0 >r000 2>r000 &
nohup python rec.py 1 375 25 0 257 64 True norand True 375 >r375 2>r375 &
nohup python rec.py 2 375 25 0 257 64 True norand True 750 >r750 2>r750 &
nohup python rec.py 3 375 25 0 257 64 True norand True 1125 >r1125 2>r1125 &

tomo5
nohup python rec.py 0 375 25 0 257 64 True norand False 0 >rn000 2>rn000 &
nohup python rec.py 1 375 25 0 257 64 True norand False 375 >nr375 2>rn375 &
nohup python rec.py 2 375 25 0 257 64 True norand False 750 >rn750 2>rn750 &
nohup python rec.py 3 375 25 0 257 64 True norand False 1125 >rn1125 2>rn1125 &



tomo4
nohup python rec.py 0 375 25 0 257 64 True rand True 0 >r000 2>r000 &
nohup python rec.py 1 375 25 0 257 64 True rand True 375 >r375 2>r375 &
nohup python rec.py 2 375 25 0 257 64 True rand True 750 >r750 2>r750 &
nohup python rec.py 3 375 25 0 257 64 True rand True 1125 >r1125 2>r1125 &

tomo5
nohup python rec.py 0 375 25 0 257 64 True rand False 0 >rn000 2>rn000 &
nohup python rec.py 1 375 25 0 257 64 True rand False 375 >nr375 2>rn375 &
nohup python rec.py 2 375 25 0 257 64 True rand False 750 >rn750 2>rn750 &
nohup python rec.py 3 375 25 0 257 64 True rand False 1125 >rn1125 2>rn1125 &
