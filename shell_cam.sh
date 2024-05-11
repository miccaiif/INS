#!/bin/bash
python train_withbag_cam.py --multiprocessing-distributed --wl 0.5
python train_withbag_cam.py --multiprocessing-distributed --wl 0
# python train_withbag_cam.py --multiprocessing-distributed --wl 1