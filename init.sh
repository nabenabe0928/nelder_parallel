#!/bin/bash
USER=$(whoami)
CWD=$(dirname $0)


rm storage/CNN/0/storage.csv
echo $USER:~$CWD$ rm storage/CNN/0/storage.csv
rm simplex/CNN/0/simplex.csv
echo $USER:~$CWD$ rm simplex/CNN/0/simplex.csv
rm operation/CNN/0/operations.csv
echo $USER:~$CWD$ rm operation/CNN/0/operations.csv
rm evaluation/CNN/0/evaluation.csv
echo $USER:~$CWD$ rm evaluation/CNN/0/evaluation.csv
rm log/CNN/0/*.csv
echo $USER:~$CWD$ rm log/CNN/0/*.csv
rm exec_screen/CNN/0/*.log
echo $USER:~$CWD$ rm exec_screen/CNN/0/*.log
