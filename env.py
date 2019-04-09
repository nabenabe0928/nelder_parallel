import csv
import math
from argparse import ArgumentParser as ArgPar

def load_storage(model, num, is_round):
    with open("storage/{}/storage.csv".format(model), "r", newline = "") as f:
        reader = csv.DictReader(f, delimiter = ",", quotechar = '"')
        var_names = reader.fieldnames

        target = {}
        post_storage = []
        no_storage = True

        for i, row in enumerate(reader):
            if i == 0:
                for var_name, value in row.items():
                    target[var_name] = value
                    
            else:
                no_storage = False
                post_storage.append({})
                for var_name, value in row.items():
                    post_storage[-1][var_name] = value
    
    if no_storage:
        with open("storage/{}/storage.csv".format(model), "w", newline = "") as f:
            writer = csv.DictWriter(f, fieldnames = var_names, delimiter = ",", quotechar = '"')
            writer.writeheader()
    else:
        renew_storage(post_storage, model)
    
    generate_shell(target, model, num, is_round)
        
def renew_storage(post_storage, model):
    var_names = [var_name for var_name in post_storage[0].keys()]

    with open("storage/{}/storage.csv".format(model), "w", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = var_names, delimiter = ",", quotechar = '"')
        writer.writeheader()

        save_row = {}

        for x in post_storage:
            for var_name, value in x.items():
                save_row[var_name] = value
            writer.writerow(save_row)

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

def generate_shell(target, model, num, is_round):
    scripts = ["#!/usr/bin/bash","USER=$(whoami)","CWD=dirname $0", ]
    enter = "\n"
    first_script = ""
    second_script = ""

    for s in scripts:
        first_script += s + enter
    
    second_script = "CUDA_VISIBLE_DEVICES=1 python train.py -model {} -num {} -round {} ".format(model, num, int(is_round))
    
    var_dist = load_dist(model)

    for var_name, value in target.items():
        dist = var_dist[var_name]
        v = convert_value_by_dist(value, dist)
        second_script += "-{} {} ".format(var_name, v)

    script = first_script + "echo $USER:~$CWD$ {} \n".format(second_script) + second_script

    with open("run.sh", "w") as f:
        f.writelines(script)

def main(model, num, is_round):
    load_storage(model, num, is_round)

if __name__ == "__main__":
    argp = ArgPar()
    argp.add_argument("-model", type = str)
    argp.add_argument("-num", type = int)
    argp.add_argument("-round", type = int, default = 1, choices = [0, 1])
    args = argp.parse_args()
    
    model = args.model
    num = args.num
    is_round = bool(args.round)

    print("Collecting Environment Variables and Putting in Shell Scripts.")
    main(model, num, is_round)