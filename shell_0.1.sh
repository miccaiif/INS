#!/bin/bash
# python train_final_11_26.py --multiprocessing-distributed --prot_start 70 --partial_rate 0.2
python train_withbag.py --multiprocessing-distributed --prot_start 20 --epoch 80
python train_withbag.py --multiprocessing-distributed --prot_start 30 --epoch 80 
python train_withbag.py --multiprocessing-distributed --prot_start 40 --epoch 80
python train_withbag.py --multiprocessing-distributed --prot_start 50 --epoch 100
python train_withbag.py --multiprocessing-distributed --prot_start 60 --epoch 100
