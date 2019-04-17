#!/bin/bash
USER=$(whoami)
CWD=$(dirname $0)


echo $USER:~$CWD$ python nelder.py -model CNN -num 0 -round 0
python nelder.py -model CNN -num 0 -round 0
echo
echo $USER:~$CWD$ python env.py -model CNN -num 0 -round 0 -cuda 1
python env.py -model CNN -num 0 -round 0 -cuda 1
echo
echo $USER:~$CWD$ ./run.sh
./run.sh
echo
