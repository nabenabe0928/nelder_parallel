#!/usr/bin/bash
USER=$(whoami)
CWD=dirname $0


rm storage/WideResNet/storage.csv
echo $USER:~$CWD$ rm storage/WideResNet/storage.csv
rm simplex/WideResNet/simplex.csv
echo $USER:~$CWD$ rm simplex/WideResNet/simplex.csv
rm operation/WideResNet/0/operations.csv
echo $USER:~$CWD$ rm operation/WideResNet/0/operations.csv
rm evaluation/WideResNet/0/evaluation.csv
echo $USER:~$CWD$ rm evaluation/WideResNet/0/evaluation.csv
rm log/WideResNet/0/*.csv
echo $USER:~$CWD$ rm log/WideResNet/0/*.csv


for i in `seq 0 100`
do
echo $USER:~$CWD$ python nelder.py -model WideResNet -num 0 -round 0
python nelder.py -model WideResNet -num 0 -round 0
echo
echo $USER:~$CWD$ python env.py -model WideResNet -num 0 -round 0
python env.py -model WideResNet -num 0 -round 0
echo
echo $USER:~$CWD$ sh run.sh
sh run.sh
echo
done
