#!/usr/bin/bash
USER=$(whoami)
CWD=dirname $0
echo $USER:~$CWD$ python train.py -model aaa -num 0 -round 0 -x1 -0.17081396241061914 -x2 -0.036864836093824106  
python train.py -model aaa -num 0 -round 0 -x1 -0.17081396241061914 -x2 -0.036864836093824106 