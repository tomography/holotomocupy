
ssh -t tomo@tomo4 "bash -c 'source ~/.bashrc; pkill -9 python;conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/experimental/AtomiumS2;ls; \\
nohup python step6_rec_iterative_levels.py configs/config6.conf > out/run6.out & bash'" 




ssh -t tomo@tomo3 "bash -c 'source ~/.bashrc; pkill -9 python; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/experimental/AtomiumS2;ls; \\
nohup python step6_rec_iterative_levels.py configs/config7.conf > out/run7.out & bash'" 


# ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc;conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/experimental/AtomiumL1;ls; \\
# nohup python step6_rec_iterative_levels.py configs/config1.conf > out/run1.out & bash'" 



# ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/experimental/AtomiumL1;ls; \\
# nohup python step6_rec_iterative_levels.py configs/config4.conf > out/run4.out & bash'" 





ssh -t tomo@tomo4 "bash -c 'source ~/.bashrc;conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/experimental/AtomiumL1;ls; \\
nohup python step6_rec_iterative_levels.py configs/config4.conf > out/run4.out & bash'" 

# ssh -t tomo@tomo4 "bash -c 'source ~/.bashrc; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/experimental/AtomiumL1;ls; \\
# nohup python step6_rec_iterative_levels.py configs/config5.conf > out/run5.out & bash'" 



ssh -t tomo@tomo3 "bash -c 'source ~/.bashrc; pkill -9 python; conda activate holotomocupy; \\
cd /home/beams/TOMO/vnikitin/holotomocupy/experimental/AtomiumL1;ls; \\
nohup python step6_rec_iterative_levels.py configs/config4.conf > out/run4.out & bash'" 

# ssh -t tomo@tomo3 "bash -c 'source ~/.bashrc; pkill -9 python; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/holotomocupy/experimental/AtomiumL1;ls; \\
# nohup python step6_rec_iterative_levels.py configs/config4.conf > out/run4.out & bash'" 




# ssh -t tomo@tomo4 "bash -c 'source ~/.bashrc; pkill -9 python; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/brainESRF/Y350a_dist1234;ls; \\
# nohup python step6_rec_iterative_levels.py configs/config6.conf > out/run6.out & bash'" 



# ssh -t tomo@tomo4 "bash -c 'source ~/.bashrc; pkill -9 python; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/brainESRF/Y350a_dist1234;ls; \\
# nohup python step6_rec_iterative_levels.py configs/config3.conf > out/run3.out & bash'" 


# ssh -t tomo@tomo2 "bash -c 'source ~/.bashrc; pkill -9 python; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/brainESRF/Y350a_dist1234;ls; \\
# nohup python step6_rec_iterative_levels.py configs/config4.conf > out/run4.out & bash'" 

# ssh -t tomo@tomo3 "bash -c 'source ~/.bashrc; pkill -9 python; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/brainESRF/Y350a_dist1234;ls; \\
# nohup python step6_rec_iterative_levels.py configs/config5.conf > out/run5.out & bash'" 

# ssh -t tomo@tomo5 "bash -c 'source ~/.bashrc; pkill -9 python; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/brainESRF/Y350a_dist1234;ls; \\
# nohup python step6_rec_iterative_levels.py 4500 2048 0 3.1e-3 20 1.1 0.05 0.02 0 > out/4500_2048_0_3.1e-3_20_1.1_0.05_0.02_0.out & bash'" 





# ssh -t tomo@tomo2 "bash -c 'source ~/.bashrc; pkill -9 python; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/brainESRF/Y350a_dist1234;ls; \\
# nohup python step6_rec_iterative_levels_nocorr.py 4500 2048 0.0 3.1e-3 20 1.1 0.05 0.02 0 > out/nocorr4500_2048_0_3.1e-3_20_1.1_0.05_0.02_0.out & bash'" 

# ssh -t tomo@tomo3 "bash -c 'source ~/.bashrc; pkill -9 python; conda activate holotomocupy; \\
# cd /home/beams/TOMO/vnikitin/brainESRF/Y350a_dist1234;ls; \\
# nohup python step6_rec_iterative_levels.py 4500 2048 0 3.2e-3 20 1.1 0.05 0.02 0 > out/objcomplex4500_2048_0_3.2e-3_20_1.1_0.05_0.02_0.out & bash'" 




# # ssh -t tomo@tomo2 "bash -c 'source ~/.bashrc; pkill -9 python; conda activate holotomocupy; \\
# # cd /home/beams/TOMO/vnikitin/brainESRF/Y350a_dist1234;ls; \\
# # nohup python step6_rec_iterative_levels_nocorr.py 4500 2048 2e-6 3.1e-3 20 1.1 0.05 0.1 0 > out/nocorr4500_2048_2e-6_3.1e-3_20_1.1_0.05_0.1_0.out & bash'" 

# watch tail -n 3 \
# out/4500_2048_2e-6_3.1e-3_20_1.1_0.05_0.02_0.out \
# out/4500_2048_0_3.1e-3_20_1.1_0.05_0.02_0.out \
# out/objcomplex4500_2048_0_3.2e-3_20_1.1_0.05_0.02_0.out



# # ssh -t tomo@tomo3 "bash -c 'source ~/.bashrc; pkill -9 python; conda activate holotomocupy; \\
# # cd /home/beams/TOMO/vnikitin/brainESRF/Y350a_dist1234;ls; \\
# # nohup python step6_rec_iterative_levels_continue.py 4500 2048 2e-6 3e-3 20 1.1 0.05 0.02 0 > out/4500_2048_2e-6_3e-3_20_1.1_0.05_0.02_0.out & bash'" 




