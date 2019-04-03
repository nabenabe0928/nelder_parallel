#!/usr/bin/bash
USER=$(whoami)
CWD=dirname $0
echo $USER:~$CWD$ python train.py -x1 -0.7137776856816821 -x2 0.20321981129237704  
python train.py -x1 -0.7137776856816821 -x2 0.20321981129237704 