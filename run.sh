#!/bin/bash
USER=$(whoami)
CWD=$(dirname $0)
echo $USER:~$CWD$ CUDA_VISIBLE_DEVICES=1 python train.py -model CNN -num 0 -round 0 -batch_size 61 -lr 0.03262505326556334 -momentum 0.939835682463644 -weight_decay 4.583944891534567e-05 -ch1 20 -ch2 25 -ch3 116 -ch4 18 -drop_rate 0.4377655731276574  
CUDA_VISIBLE_DEVICES=1 python train.py -model CNN -num 0 -round 0 -batch_size 61 -lr 0.03262505326556334 -momentum 0.939835682463644 -weight_decay 4.583944891534567e-05 -ch1 20 -ch2 25 -ch3 116 -ch4 18 -drop_rate 0.4377655731276574 