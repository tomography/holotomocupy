
python rec.py 1 50 5 2 129 64 >new50_5_2 2>new50_5_2e
tomo1
nohup python rec.py 1 1500 5 0 4097 64 >new1500_5_0 2>new1500_5_0e &
tomo2
nohup python rec.py 1 750 5 0 4097 64 >new750_5_0 2>new750_5_0e &

tomo4
nohup python rec.py 1 750 20 1 4097 64 >new750_20_1 2>new750_20_1e &
tomo4
nohup python rec.py 2 300 20 1 4097 64 >new300_20_1 2>new300_20_1e &
tomo4
nohup python rec.py 3 150 20 1 4097 64 >new150_20_1 2>new150_20_1e &

tomo5
nohup python rec.py 1 300 5 0 4097 64 >new300_5_0 2>new300_5_0e &

nohup python rec.py 2 150 5 0 4097 64 >new150_5_0 2>new150_5_0e &

nohup python rec.py 3 50 5 0 4097 64 >new50_5_0 2>new50_5_0e &



new tomo3


tomo2
nohup python rec.py 0 150 20 1 4097 64 >new150_20_1 2>new150_20_1e &

tomo3
nohup python rec.py 0 5 5 0 4097 64 4 >nnew5_5_0_4 2>nnew5_5_0_4e &
nohup python rec.py 0 25 5 0 4097 64 3 >nnew25_5_0_3 2>nnew25_5_0_3e &
nohup python rec.py 0 25 5 0 4097 64 2 >nnew25_5_0_2 2>nnew25_5_0_2e &
nohup python rec.py 0 25 5 0 4097 64 1 >nnew25_5_0_1 2>nnew25_5_0_1e &

tomo4
nohup python rec.py 0 25 5 0 4097 64 4 >nnew25_5_0_4 2>nnew25_5_0_4e &
nohup python rec.py 3 10 5 0 4097 64 4 >nnew10_5_0_4 2>nnew10_5_0_4e &


# tomo5
# nohup python rec.py 0 300 10 1 4097 64 3 >nnew300_10_1_3 2>nnew300_10_1_3e &
# nohup python rec.py 1 300 10 1 4097 64 2 >nnew300_10_1_2 2>nnew300_10_1_2e &
# nohup python rec.py 2 300 10 1 4097 64 1 >nnew300_10_1_1 2>nnew300_10_1_1e &
# nohup python rec.py 3 750 10 1 4097 64 3 >nnew750_10_1_3 2>nnew750_10_1_3e &


tomo5 
nohup python rec.py 0 100 5 0 2048 64 4 >100_5_0_4 2>100_5_0_4e &
nohup python rec.py 1 10 5 0 2048 64 4 >10_5_0_4 2>10_5_0_4e &
nohup python rec.py 2 25 5 0 2048 64 4 >25_5_0_4 2>25_5_0_4e &
nohup python rec.py 3 50 5 0 2048 64 4 >50_5_0_4 2>50_5_0_4e &

tomo4
nohup python rec.py 0 150 5 0 2048 64 4 >150_5_0_4 2>150_5_0_4e &
nohup python rec.py 1 150 5 0 2048 64 3 >150_5_0_3 2>150_5_0_3e &
nohup python rec.py 2 150 5 0 2048 64 2 >150_5_0_2 2>150_5_0_2e &
nohup python rec.py 3 150 5 0 2048 64 1 >150_5_0_1 2>150_5_0_1e &

tomo3
nohup python rec.py 0 300 5 0 2048 64 4 >300_5_0_4 2>300_5_0_4e &


tomo2
nohup python rec.py 1 500 20 1 1025 64 4 >500_20_1_4 2>500_20_1_4e &


tomo3
nohup python rec_parts.py 0 300 5 0 2048 64 4 0 >300_5_0_4_0 2>300_5_0_4_0e &

tomo4
nohup python rec_parts.py 0 150 5 0 2048 64 4 0 >150_5_0_4_0 2>150_5_0_4_0e &
nohup python rec_parts.py 1 150 5 0 2048 64 4 150 >150_5_0_4_150 2>150_5_0_4_150e &
nohup python rec_parts.py 2 150 5 0 2048 64 4 300 >150_5_0_4_300 2>150_5_0_4_300e &
nohup python rec_parts.py 3 150 5 0 2048 64 4 450 >150_5_0_4_450 2>150_5_0_4_450e &


tomo5
nohup python rec_parts.py 0 150 5 0 2048 64 4 600 >150_5_0_4_600 2>150_5_0_4_600e &
nohup python rec_parts.py 1 150 5 0 2048 64 4 750 >150_5_0_4_750 2>150_5_0_4_750e &
nohup python rec_parts.py 2 150 5 0 2048 64 4 900 >150_5_0_4_900 2>150_5_0_4_900e &
nohup python rec_parts.py 3 150 5 0 2048 64 4 1050 >150_5_0_4_1050 2>150_5_0_4_1050e &


tomo2
nohup python rec_parts.py 0 150 5 0 2048 64 4 1200 >150_5_0_4_1200 2>150_5_0_4_1200e &
nohup python rec_parts.py 1 150 5 0 2048 64 4 1350 >150_5_0_4_1350 2>150_5_0_4_1350e &



tomo5
nohup python rec_parts2.py 0 150 5 0 2048 64 4 600 >same_probe150_5_0_4_600 2>same_probe150_5_0_4_600e &
nohup python rec_parts2.py 1 150 5 0 2048 64 4 750 >same_probe150_5_0_4_750 2>same_probe150_5_0_4_750e &
nohup python rec_parts2.py 2 150 5 0 2048 64 4 900 >same_probe150_5_0_4_900 2>same_probe150_5_0_4_900e &
nohup python rec_parts2.py 3 150 5 0 2048 64 4 1050 >same_probe150_5_0_4_1050 2>same_probe150_5_0_4_1050e &

tomo4
nohup python rec_parts2.py 0 150 5 0 2048 64 4 0 >same_probe150_5_0_4_0 2>same_probe150_5_0_4_0e &
nohup python rec_parts2.py 1 150 5 0 2048 64 4 150 >same_probe150_5_0_4_150 2>same_probe150_5_0_4_150e &
nohup python rec_parts2.py 2 150 5 0 2048 64 4 300 >same_probe150_5_0_4_300 2>1same_probe50_5_0_4_300e &
nohup python rec_parts2.py 3 150 5 0 2048 64 4 450 >same_probe150_5_0_4_450 2>same_probe150_5_0_4_450e &

tomo1
nohup python rec_parts2.py 0 150 5 0 2048 64 4 1200 >same_probe150_5_0_4_1200 2>same_probe150_5_0_4_1200e &
nohup python rec_parts2.py 1 150 5 0 2048 64 4 1350 >same_probe150_5_0_4_1350 2>same_probe150_5_0_4_1350e &




nohup python rec_parts2.py 1 150 5 2 2048 32 4 1350 >150_5_2_4_1350 2>150_5_2_4_1350e &







cp.cuda.Device(int(sys.argv[1])).use()
ntheta = int(sys.argv[2])#750  # number of angles (rotations)
ptheta = int(sys.argv[3])  # holography chunk size for GPU processing
binning = int(sys.argv[4])
st =  int(sys.argv[5])
niter = int(sys.argv[6])
iter_step = int(sys.argv[7])


tomo4
nohup python rec_peter_parts.py 0 150 10 0 8193 64 4 0 >same_probe150_10_0_4_0 2>same_probe150_10_0_4_0 &
nohup python rec_peter_parts.py 1 150 10 0 8193 64 4 150 >same_probe150_10_0_4_150 2>same_probe150_10_0_4_150 &
nohup python rec_peter_parts.py 2 150 10 0 8193 64 4 300 >same_probe150_10_0_4_300 2>same_probe150_10_0_4_300 &
nohup python rec_peter_parts.py 3 150 10 0 8193 64 4 450 >same_probe150_10_0_4_450 2>same_probe150_10_0_4_450 &


tomo5
nohup python rec_peter_parts.py 0 150 10 0 8193 64 4 600 >same_probe150_10_0_4_600 2>same_probe150_10_0_4_600 &
nohup python rec_peter_parts.py 1 150 10 0 8193 64 4 750 >same_probe150_10_0_4_750 2>same_probe150_10_0_4_750 &
nohup python rec_peter_parts.py 2 150 10 0 8193 64 4 900 >same_probe150_10_0_4_900 2>same_probe150_10_0_4_900 &
nohup python rec_peter_parts.py 3 150 10 0 8193 64 4 1050 >same_probe150_10_0_4_1050 2>same_probe150_10_0_4_1050 &

tomo1
nohup python rec_peter_parts.py 0 150 10 0 8193 64 4 1200 >same_probe150_10_0_4_1200 2>same_probe150_10_0_4_1200 &
nohup python rec_peter_parts.py 0 150 10 0 8193 64 4 1350 >same_probe150_10_0_4_1350 2>same_probe150_10_0_4_1350 &


tomo1

nohup python rec_peter_parts.py 0 1500 10 0 8193 32 4 0 >same_probe1500_10_0_4_0 2>same_probe1500_10_0_4_0 &
nohup python rec_peter_parts.py 0 1500 10 0 8193 32 4 0 >same_probeFalse1500_10_0_4_0 2>same_probeFalse1500_10_0_4_0 &


tomo5
nohup python rec_peter_parts.py 0 375 15 1 8193 64 4 0 True >same_probeTrue375_25_1_4_0 2>same_probeTrue375_25_1_4_0 &
nohup python rec_peter_parts.py 1 375 15 1 8193 64 4 375 True >same_probeTrue375_25_1_4_375 2>same_probeTrue375_25_1_4_375 &
nohup python rec_peter_parts.py 2 375 15 1 8193 64 4 750 True >same_probeTrue375_25_1_4_750 2>same_probeTrue375_25_1_4_750 &
nohup python rec_peter_parts.py 3 375 15 1 8193 64 4 1125 True >same_probeTrue375_25_1_4_1125 2>same_probeTrue375_25_1_4_1125 &


nohup python rec_peter_parts.py 0 375 15 0 8193 64 4 0 False >same_probeFalse375_15_0_4_0 2>same_probeFalse375_15_0_4_0 &
nohup python rec_peter_parts.py 1 375 15 0 8193 64 4 375 False >same_probeFalse375_15_0_4_375 2>same_probeFalse375_15_0_4_375 &
nohup python rec_peter_parts.py 2 375 15 0 8193 64 4 750 False >same_probeFalse375_15_0_4_750 2>same_probeFalse375_15_0_4_750 &
nohup python rec_peter_parts.py 3 375 15 0 8193 64 4 1125 False >same_probeFalse375_15_0_4_1125 2>same_probeFalse375_15_0_4_1125 &

tomo3
nohup python rec_peter_parts.py 0 1500 15 0 8193 64 4 0 True >same_probeTrue1500_15_0_4_0 2>same_probeTrue1500_15_0_4_0 &
tomo1
nohup python rec_peter_parts.py 0 1500 15 0 8193 64 4 0 False >same_probeFalse1500_15_0_4_0 2>same_probeFalse1500_15_0_4_0 &

tomo4
nohup python rec_peter_parts.py 0 750 5 0 8193 64 4 0 False >same_probeFalse750_5_0_4_0 2>same_probeFalse750_5_0_4_0 &
nohup python rec_peter_parts.py 1 750 5 0 8193 64 4 750 False >same_probeFalse750_5_0_4_750 2>same_probeFalse750_5_0_4_750 &

tomo2
nohup python rec_peter_parts.py 0 750 5 0 8193 64 4 0 True >same_probeFalse750_5_0_4_0 2>same_probeTrue750_5_0_4_0 &
nohup python rec_peter_parts.py 1 750 5 0 8193 64 4 750 True >same_probeFalse750_5_0_4_750 2>same_probeTrue750_5_0_4_750 &


tomo3
nohup python rec_peter_parts.py 0 750 15 0 8193 64 4 0 True >same_probeTrue750_15_0_4_0 2>same_probeTrue750_15_0_4_0 &
tomo1
nohup python rec_peter_parts.py 0 750 15 0 8193 64 4 750 True >same_probeTrue750_15_0_4_750 2>same_probeTrue750_15_0_4_750 &


tomo5
nohup python rec_peter_parts.py 0 375 25 1 8193 64 4 0 True >same_probeTrue375_25_1_4_0 2>same_probeTrue375_25_1_4_0 &
nohup python rec_peter_parts.py 1 375 25 1 8193 64 4 375 True >same_probeTrue375_25_1_4_375 2>same_probeTrue375_25_1_4_375 &
nohup python rec_peter_parts.py 2 375 25 1 8193 64 4 750 True >same_probeTrue375_25_1_4_750 2>same_probeTrue375_25_1_4_750 &
nohup python rec_peter_parts.py 3 375 25 1 8193 64 4 1125 True >same_probeTrue375_25_1_4_1125 2>same_probeTrue375_25_1_4_1125 &


tomo4
nohup python rec_peter_parts.py 0 375 25 1 8193 64 4 0 False >same_probeFalse375_25_1_4_0 2>same_probeFalse375_25_1_4_0 &
nohup python rec_peter_parts.py 1 375 25 1 8193 64 4 375 False >same_probeFalse375_25_1_4_375 2>same_probeFalse375_25_1_4_375 &
nohup python rec_peter_parts.py 2 375 25 1 8193 64 4 750 False >same_probeFalse375_25_1_4_750 2>same_probeFalse375_25_1_4_750 &
nohup python rec_peter_parts.py 3 375 25 1 8193 64 4 1125 False >same_probeFalse375_25_1_4_1125 2>same_probeFalse375_25_1_4_1125 &

tomo3
nohup python rec_peter_parts.py 0 1500 50 1 8193 64 4 0 True >same_probeTrue1500_50_1_4_0 2>same_probeTrue1500_50_1_4_0 &

tomo2
nohup python rec_peter_parts.py 0 750 25 1 8193 64 4 0 True >same_probeTrue750_25_1_4_0 2>same_probeTrue750_25_1_4_0 &
nohup python rec_peter_parts.py 1 750 25 1 8193 64 4 0 False >same_probeFalse750_25_1_4_0 2>same_probeFalse750_25_1_4_0 &

tomo1
nohup python rec_peter_parts.py 0 150 25 1 8193 64 4 0 True >same_probeTrue150_25_1_4_0 2>same_probeTrue150_25_1_4_0 &
nohup python rec_peter_parts.py 0 150 25 1 8193 64 4 0 False >same_probeFalse150_25_1_4_0 2>same_probeFalse150_25_1_4_0 &


tomo5
nohup python rec_peter_parts.py 0 50 25 1 18193 64 4 0 True >same_probeTrue50_25_1_4_0 2>same_probeTrue50_25_1_4_0 &
nohup python rec_peter_parts.py 1 100 25 1 18193 64 4 0 True >same_probeTrue100_25_1_4_0 2>same_probeTrue100_25_1_4_0 &
nohup python rec_peter_parts.py 2 150 25 1 18193 64 4 0 True >same_probeTrue150_25_1_4_0 2>same_probeTrue150_25_1_4_0 &
nohup python rec_peter_parts.py 3 375 25 1 18193 64 4 0 True >same_probeTrue375_25_1_4_0 2>same_probeTrue375_25_1_4_0 &

tomo4
nohup python rec_peter_parts.py 0 50 25 1 18193 64 4 0 False >same_probeFalse50_25_1_4_0 2>same_probeFalse50_25_1_4_0 &
nohup python rec_peter_parts.py 1 100 25 1 18193 64 4 0 False >same_probeFalse100_25_1_4_0 2>same_probeFalse100_25_1_4_0 &
nohup python rec_peter_parts.py 2 150 25 1 18193 64 4 0 False >same_probeFalse150_25_1_4_0 2>same_probeFalse150_25_1_4_0 &
nohup python rec_peter_parts.py 3 375 25 1 18193 64 4 0 False >same_probeFalse375_25_1_4_0 2>same_probeFalse375_25_1_4_0 &

tomo2 
nohup python rec_peter_parts.py 0 750 25 1 18193 64 4 0 True >same_probeTrue750_25_1_4_0 2>same_probeTrue750_25_1_4_0 &
nohup python rec_peter_parts.py 1 750 25 1 18193 64 4 0 False >same_probeFalse750_25_1_4_0 2>same_probeFalse750_25_1_4_0 &

tomo3
nohup python rec_peter_parts.py 0 1500 50 1 18193 64 4 0 True >same_probeTrue1500_50_1_4_0 2>same_probeTrue1500_50_1_4_0 &

tomo1
nohup python rec_peter_parts.py 0 1500 25 1 18193 64 4 0 False >same_probeFalse1500_25_1_4_0 2>same_probeFalse1500_25_1_4_0 &


tomo5
nohup python rec_peter_parts.py 0 75 25 1 18193 64 4 0 True >same_probeTrue75_25_1_4_0 2>same_probeTrue75_25_1_4_0 &


tomo4
nohup python rec_peter_parts.py 0 50 50 2 18193 128 4 0 True >same_probeTrue50_50_2_4_0 2>same_probeTrue50_50_2_4_0 &
nohup python rec_peter_parts.py 1 100 100 2 18193 128 4 0 True >same_probeTrue100_100_2_4_0 2>same_probeTrue100_100_2_4_0 &
nohup python rec_peter_parts.py 2 150 150 2 18193 128 4 0 True >same_probeTrue150_150_2_4_0 2>same_probeTrue150_150_2_4_0 &
nohup python rec_peter_parts.py 3 375 375 2 18193 128 4 0 True >same_probeTrue375_375_2_4_0 2>same_probeTrue375_375_2_4_0 &
nohup python rec_peter_parts.py 0 75 75 2 18193 128 4 0 True >same_probeTrue75_75_2_4_0_sec_prb 2>same_probeTrue75_75_2_4_0_sec_prb &

nohup python rec_peter_parts.py 0 50 50 2 18193 128 4 0 False >same_probeFalse50_50_2_4_0 2>same_probeFalse50_50_2_4_0 &
nohup python rec_peter_parts.py 1 100 100 2 18193 128 4 0 False >same_probeFalse100_100_2_4_0 2>same_probeFalse100_100_2_4_0 &
nohup python rec_peter_parts.py 2 150 150 2 18193 128 4 0 False >same_probeFalse150_150_2_4_0 2>same_probeFalse150_150_2_4_0 &
nohup python rec_peter_parts.py 3 375 375 2 18193 128 4 0 False >same_probeFalse375_375_2_4_0 2>same_probeFalse375_375_2_4_0 &
nohup python rec_peter_parts.py 0 75 75 2 18193 128 4 0 False >same_probeFalse75_75_2_4_0_sec_prb 2>same_probeFalse75_75_2_4_0_sec_prb &



tomo4
nohup python rec_peter_parts.py 0 250 25 1 8193 64 4 0 True >same_probeTrue250_25_1_4_0 2>same_probeTrue250_25_1_4_0 &
nohup python rec_peter_parts.py 1 250 25 1 8193 64 4 1 True >same_probeTrue250_25_1_4_1 2>same_probeTrue250_25_1_4_1 &
nohup python rec_peter_parts.py 2 250 25 1 8193 64 4 2 True >same_probeTrue250_25_1_4_2 2>same_probeTrue250_25_1_4_2 &
nohup python rec_peter_parts.py 3 250 25 1 8193 64 4 3 True >same_probeTrue250_25_1_4_3 2>same_probeTrue250_25_1_4_3 &

tomo3
nohup python rec_peter_parts.py 0 250 25 1 8193 64 4 4 True >same_probeTrue250_25_1_4_2 2>same_probeTrue250_25_1_4_4 &

nohup python rec_peter_parts.py 0 250 25 1 8193 64 4 5 True >same_probeTrue250_25_1_4_3 2>same_probeTrue250_25_1_4_5 &

tomo5
nohup python rec_peter_parts.py 0 250 25 1 8193 64 4 0 False >same_probeFalse250_25_1_4_0 2>same_probeFalse250_25_1_4_0 &
nohup python rec_peter_parts.py 1 250 25 1 8193 64 4 1 False >same_probeFalse250_25_1_4_1 2>same_probeFalse250_25_1_4_1 &
nohup python rec_peter_parts.py 2 250 25 1 8193 64 4 2 False >same_probeFalse250_25_1_4_2 2>same_probeFalse250_25_1_4_2 &
nohup python rec_peter_parts.py 3 250 25 1 8193 64 4 3 False >same_probeFalse250_25_1_4_3 2>same_probeFalse250_25_1_4_3 &


tomo2
nohup python rec_peter_parts.py 0 250 25 1 8193 64 4 4 False >same_probeFalse250_25_1_4_2 2>same_probeFalse250_25_1_4_4 &
nohup python rec_peter_parts.py 1 250 25 1 8193 64 4 5 False >same_probeFalse250_25_1_4_3 2>same_probeFalse250_25_1_4_5 &



tomo4
nohup python rec_peter_parts.py 0 125 5 0 8193 64 4 0 True >same_probeTrue125_5_0_4_0 2>same_probeTrue125_5_0_4_0 &
nohup python rec_peter_parts.py 1 125 5 0 8193 64 4 1 True >same_probeTrue125_5_0_4_1 2>same_probeTrue125_5_0_4_1 &
nohup python rec_peter_parts.py 2 125 5 0 8193 64 4 2 True >same_probeTrue125_5_0_4_2 2>same_probeTrue125_5_0_4_2 &
nohup python rec_peter_parts.py 3 125 5 0 8193 64 4 3 True >same_probeTrue125_5_0_4_3 2>same_probeTrue125_5_0_4_3 &

tomo5
nohup python rec_peter_parts.py 0 150 10 0 10001 64 4 4 True >paganin_probeTrue150_10_0_4_4 2>paganin_probeTrue150_10_0_4_4 &
nohup python rec_peter_parts.py 0 150 10 0 10001 64 4 4 True >paganin_probeTrue150_10_0_4_4 2>paganin_probeTrue150_10_0_4_4 &

nohup python rec_peter_parts.py 1 150 5 0 10001 64 4 5 True >same_probeTrue125_5_0_4_5 2>same_probeTrue125_5_0_4_5 &
nohup python rec_peter_parts.py 2 150 5 0 10001 64 4 6 True >same_probeTrue125_5_0_4_6 2>same_probeTrue125_5_0_4_6 &
nohup python rec_peter_parts.py 3 150 5 0 10001 64 4 7 True >same_probeTrue125_5_0_4_7 2>same_probeTrue125_5_0_4_7 &

tomo2
nohup python rec_peter_parts.py 0 125 5 0 8193 64 4 8 True >same_probeTrue125_5_0_4_8 2>same_probeTrue125_5_0_4_8 &
nohup python rec_peter_parts.py 1 125 5 0 8193 64 4 9 True >same_probeTrue125_5_0_4_9 2>same_probeTrue125_5_0_4_9 &

nohup python rec_peter_parts.py 0 125 5 0 8193 64 4 10 True >same_probeTrue125_5_0_4_10 2>same_probeTrue125_5_0_4_10 &

nohup python rec_peter_parts.py 0 125 5 0 8193 64 4 11 True >same_probeTrue125_5_0_4_11 2>same_probeTrue125_5_0_4_11 &



tomo2

nohup python rec_peter_parts.py 1 1500 10 0 8193 64 4 0 True >same_probeTrue1500_10_0_4_0 2>same_probeTrue1500_10_0_4_0 &

tomo1
nohup python rec_peter_parts.py 1 1500 10 0 8193 64 4 0 False >same_probeFalse1500_10_0_4_0 2>same_probeFalse1500_10_0_4_0 &


tomo5
nohup python rec_peter_parts.py 0 150 10 0 10001 64 4 0 True >paganin_probeTrue150_10_0_4_0 2>paganin_probeTrue150_10_0_4_0 &
nohup python rec_peter_parts.py 1 150 10 0 10001 64 4 1 True >paganin_probeTrue150_10_0_4_1 2>paganin_probeTrue150_10_0_4_1 &
nohup python rec_peter_parts.py 2 150 10 0 10001 64 4 2 True >paganin_probeTrue150_10_0_4_2 2>paganin_probeTrue150_10_0_4_2 &
nohup python rec_peter_parts.py 3 150 10 0 10001 64 4 3 True >paganin_probeTrue150_10_0_4_3 2>paganin_probeTrue150_10_0_4_3 &

tomo2
nohup python rec_peter_parts.py 0 150 10 0 10001 64 4 4 True >paganin_probeTrue150_5_0_4_4 2>paganin_probeTrue150_5_0_4_4 &
nohup python rec_peter_parts.py 1 150 10 0 10001 64 4 5 True >paganin_probeTrue150_5_0_4_5 2>paganin_probeTrue150_5_0_4_5 &
nohup python rec_peter_parts.py 0 150 10 0 10001 64 4 6 True >paganin_probeTrue150_5_0_4_6 2>paganin_probeTrue150_5_0_4_6 &
nohup python rec_peter_parts.py 1 150 10 0 10001 64 4 7 True >paganin_probeTrue150_5_0_4_7 2>paganin_probeTrue150_5_0_4_7 &

tomo3
nohup python rec_peter_parts.py 0 150 5 0 10001 64 4 8 True >paganin_probeTrue150_5_0_4_8 2>paganin_probeTrue150_5_0_4_8 &
nohup python rec_peter_parts.py 0 150 5 0 10001 64 4 9 True >paganin_probeTrue150_5_0_4_9 2>paganin_probeTrue150_5_0_4_9 &

tomo2
nohup python rec_final_noprobe.py 0 250 5 0 100 128 4 4 True >final_noprobe_4 2>final_noprobe_250_4 &
nohup python rec_final_noprobe.py 1 250 5 0 100 128 4 5 True >final_noprobe_5 2>final_noprobe_250_5 &
tomo3
nohup python rec_final_noprobe.py 3 250 5 0 100 128 4 3 True >final_noprobe_3 2>final_noprobe_250_3 &
tomo5
nohup python rec_final_noprobe.py 0 250 25 0 100 128 4 0 True >final_noprobe_0 2>final_noprobe_250_0 &
nohup python rec_final_noprobe.py 1 250 25 0 100 128 4 1 True >final_noprobe_1 2>final_noprobe_250_1 &
nohup python rec_final_noprobe.py 2 250 25 0 100 128 4 2 True >final_noprobe_2 2>final_noprobe_250_2 &




