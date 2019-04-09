from argparse import ArgumentParser as ArgPar
import subprocess as sp
import sys
import math
import os

if __name__ == "__main__":
    try:
        argp = ArgPar()
        argp.add_argument("-model", type = str)
        argp.add_argument("-num", type = int)
        argp.add_argument("-itr", type = int, default = 60)
        argp.add_argument("-round", type = int, default = 1, choices = [0, 1])
        args = argp.parse_args()
        
        model = args.model
        num = args.num
        itr = args.itr
        round = bool(args.round)
        
        # to know whether you put at least -model and -num
        sys.argv[5]

    except:
        print("SET the variables shown as below:")
        print("model: which model you run: [WideResNet]")
        print("  num: how many times this experiment is: [0, 1, 2, ...]")
        print("  itr: how many times evaluating the model: [any integer]")
        print("round: whether you round up the hyperparameter when one of them is out of boundaries.")
        print("       1: Round up and Evaluate")
        print("       0: Not Round up and Record -inf as output.")
        sys.exit()
    
    if not os.path.isdir("exec_screen/{}".format(model)):
        os.mkdir("exec_screen/{}".format(model))

    sh_lines = \
        ["#!/usr/bin/bash", \
        "USER=$(whoami)", \
        "CWD=dirname $0", \
        "\n", \
        "rm storage/{}/storage.csv".format(model), \
        "echo $USER:~$CWD$ rm storage/{}/storage.csv".format(model), \
        "rm simplex/{}/simplex.csv".format(model), \
        "echo $USER:~$CWD$ rm simplex/{}/simplex.csv".format(model), \
        "rm operation/{}/{}/operations.csv".format(model, num), \
        "echo $USER:~$CWD$ rm operation/{}/{}/operations.csv".format(model, num), \
        "rm evaluation/{}/{}/evaluation.csv".format(model, num), \
        "echo $USER:~$CWD$ rm evaluation/{}/{}/evaluation.csv".format(model, num), \
        "rm log/{}/{}/*.csv".format(model, num), \
        "echo $USER:~$CWD$ rm log/{}/{}/*.csv".format(model,num), \
        "\n", \
        "for i in `seq 0 {}`".format(itr), \
        "do", \
        "echo $USER:~$CWD$ python nelder.py -model {} -num {} -round {}".format(model, num, int(round)), \
        "python nelder.py -model {} -num {} -round {}".format(model, num, int(round)), \
        "echo",  \
        "echo $USER:~$CWD$ python env.py -model {} -num {} -round {}".format(model, num, int(round)), \
        "python env.py -model {} -num {} -round {}".format(model, num, int(round)), \
        "echo",  \
        "echo $USER:~$CWD$ sh run.sh", \
        "sh run.sh", \
        "echo",  \
        "done"]
    
    script = ""

    for line in sh_lines:
        script += line + "\n"

    with open("main.sh", "w") as f:
        f.writelines(script)
    
    try:
        sp.call("rm exec_screen/{}/exec{}.log".format(model, num), shell = True)
    except:
        print("no such file exec_screen/{}/exec{}.log".format(model, num))

    sp.call("sh main.sh > exec_screen/{}/exec{}.log".format(model, num), shell = True)