#!/bin/bash
#source /home/vvnikitin/anaconda3/etc/profile.d/conda.sh
#conda activate tomoalign

case $1 in 
	0)
	python admm.py 500 sift 190 1
	;;
	1)
	python admm.py 500 sift 190 0
	;;
	2)
	python admm.py 500 stxm 190 1
	;;
	3)
	python admm.py 500 stxm 190 0
	;;
	4)
	python admm.py 3500 sift 190 1
	;;
	5)
	python admm.py 3500 sift 190 0
	;;
	6)
	python admm.py 3500 stxm 190 1
	;;
	7)
	python admm.py 3500 stxm 190 0
	;;
	8)
	python admm.py 3850 sift 190 1
	;;
	9)
	python admm.py 3850 sift 190 0
	;;
	10)
	python admm.py 1000 sift 190 1
	;;
	11)
	python admm.py 1000 sift 190 0
	;;
	12)
	python admm.py 2000 sift 190 1
	;;
	13)
	python admm.py 2000 sift 190 0
	;;

esac
	

#python admm.py 500 sift 190 0

#python admm.py 500 stxm 190 1
#python admm.py 500 stxm 190 0

#python admm.py 3500 sift 190 1
#python admm.py 3500 sift 190 0

#python admm.py 3500 stxm 190 1
#python admm.py 3500 stxm 190 0









