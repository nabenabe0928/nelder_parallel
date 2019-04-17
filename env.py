import csv
import math
import subprocess as sp
from argparse import ArgumentParser as ArgPar

def load_storage(model, num, is_round, cuda):
    with open("storage/{}/{}/storage.csv".format(model, num), "r", newline = "") as f:
        reader = csv.DictReader(f, delimiter = ",", quotechar = '"')
        target = {}
        
        for i, row in enumerate(reader):
            if i == 0:
                for var_name, value in row.items():
                    target[var_name] = value
    
    generate_shell(target, model, num, is_round, cuda)

def load_dist(model):
    with open("type_dict/{}/type_dict.csv".format(model), "r", newline = "") as f:
        reader = csv.DictReader(f, delimiter = ";", quotechar = '"')
        
        var_dist = {}

        for row in reader:
            var_dist[row["var_name"]] = [eval(row["type"]), row["dist"]]

    return var_dist

def convert_value_by_dist(value, dist):
    vtype, vdist = dist[0], dist[1]
    if "log" in vdist:
        base = float(vdist.split("log")[-1])
        v = vtype(base ** float(value))
    else:
        v = value

    return v

def generate_shell(target, model, num, is_round, cuda):
    scripts = ["#!/bin/bash","USER=$(whoami)","CWD=$(dirname $0)", ]
    enter = "\n"
    first_script = ""
    second_script = ""

    for s in scripts:
        first_script += s + enter
    
    second_script = "CUDA_VISIBLE_DEVICES={} python train.py -model {} -num {} -round {} ".format(cuda, model, num, int(is_round))
    
    var_dist = load_dist(model)

    for var_name, value in target.items():
        dist = var_dist[var_name]
        v = convert_value_by_dist(value, dist)
        second_script += "-{} {} ".format(var_name, v)

    script = first_script + "echo $USER:~$CWD$ {} \n".format(second_script) + second_script

    with open("run.sh", "w") as f:
        f.writelines(script)
    sp.call("ls", shell = True)
    sp.call("chmod +x run.sh", shell = True)

def main(model, num, is_round, cuda):
    load_storage(model, num, is_round, cuda)

if __name__ == "__main__":
    argp = ArgPar()
    argp.add_argument("-model", type = str)
    argp.add_argument("-num", type = int)
    argp.add_argument("-round", type = int, default = 1, choices = [0, 1])
    argp.add_argument("-cuda", type = int, default = 1, required = True)
    args = argp.parse_args()
    
    model = args.model
    num = args.num
    is_round = bool(args.round)
    cuda = args.cuda

    print("Collecting Environment Variables and Putting in Shell Scripts.")
    main(model, num, is_round, cuda)