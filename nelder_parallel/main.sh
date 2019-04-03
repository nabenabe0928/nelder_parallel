#!/usr/bin/bash
USER=$(whoami)
CWD=dirname $0


rm storage/aaa/storage.csv
echo $USER:~$CWD$ rm storage/aaa/storage.csv
rm simplex/aaa/simplex.csv
echo $USER:~$CWD$ rm simplex/aaa/simplex.csv
rm operation/aaa/0/operations.csv
echo $USER:~$CWD$ rm operation/aaa/0/operations.csv
rm evaluation/aaa/0/evaluation.csv
echo $USER:~$CWD$ rm evaluation/aaa/0/evaluation.csv


for i in `seq 0 20`
do
echo $USER:~$CWD$ python nelder.py -model aaa -num 0 -round 0
python nelder.py -model aaa -num 0 -round 0
echo
echo $USER:~$CWD$ python env.py -model aaa -num 0 -round 0
python env.py -model aaa -num 0 -round 0
echo
echo $USER:~$CWD$ sh run.sh
sh run.sh
echo
done
