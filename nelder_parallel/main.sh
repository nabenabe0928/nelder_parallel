#!/usr/bin/bash
USER=$(whoami)
CWD=dirname $0

rm storage/storage.csv
echo $USER:~${CWD}$ rm storage/storage.csv
rm simplex/simplex.csv
echo $USER:~${CWD}$ rm simplex/simplex.csv
rm operation/0/operations.csv
echo $USER:~${CWD}$ rm operation/0/operations.csv
rm evaluation/0/evaluation.csv
echo $USER:~${CWD}$ rm evaluation/evaluation.csv

for i in `seq 0 20`
do
echo $USER:~${CWD}$ python nelder.py
python nelder.py
echo 
echo $USER:~${CWD}$ python env.py
python env.py
echo 
echo $USER:~${CWD}$ sh run.sh
sh run.sh
echo 
done