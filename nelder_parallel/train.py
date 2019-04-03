import csv
import os
import numpy as np
from argparse import ArgumentParser as ArgPar
import sys

def load_var_features(model, is_round = True):
    with open("type_dict/{}/type_dict.csv".format(model), "r", newline = "") as f:
        reader = csv.DictReader(f, delimiter = ";", quotechar = '"')
        
        var_names, type_dict, bounds, defaults, helps = [],[],[],[],[]

        for row in reader:
            var_names.append(row["var_name"])
            type_dict.append(eval(row["type"]))
            bounds.append(eval(row["bound"]))
            defaults.append(type_dict[-1](row["default"]))
            helps.append(row["help"])

        HyperParameters, out_of_bound = get_arguments(var_names, type_dict, bounds, defaults, helps, is_round)

        return var_names, HyperParameters, out_of_bound

def get_arguments(var_names, type_dict, bounds, defaults, helps, is_round = True):
    argp = ArgPar()
    n_vars = len(var_names) - 1

    for i in range(n_vars):
        argp.add_argument("-{}".format(var_names[i]), type = type_dict[i], default = defaults[i], help = helps[i])
    
    parsed_parameters = argp.parse_args(sys.argv[7:])
    HyperParameters = {}
    out_of_bound = False

    for var_name, bound in zip(var_names[:-1], bounds[:-1]):
        hp = getattr(parsed_parameters, var_name)
        
        if is_round:
            HyperParameters[var_name] = np.clip(hp, bound[0], bound[1])
        else:
            HyperParameters[var_name] = hp

            if not (bound[0] <= hp <= bound[1]):
                out_of_bound = True

    return HyperParameters, out_of_bound


def renew_evaluation(var_names, x, y, model, num = 0):
    
    with open("evaluation/{}/{}/evaluation.csv".format(model, num), "a", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = var_names, delimiter = ",", quotechar = '"')
        
        save_row = {var_name: value for var_name, value in zip(var_names, np.append(x, y))}
        writer.writerow(save_row)
        
class function():
    def __init__(self, model, num = 0, is_round = True):
        self.num = num
        self.model = model
        self.var_names, self.HyperParameters, self.out_of_bound = load_var_features(self.model, is_round = is_round)
        self.xs = np.array([self.HyperParameters["x1"], self.HyperParameters["x2"]])
        self.f()

    def f(self):
        if not self.out_of_bound:
            y = np.sum(self.xs ** 2)
            print("Output: {:.3f}".format(y))
            print("")
        else:
            y = 1.0e+08
            print("Output: {:.3f}".format(y))
            print("")

        renew_evaluation(self.var_names, self.xs, y, self.model, num = self.num)

if __name__ == "__main__":
    argp = ArgPar()
    argp.add_argument("-model", type = str)
    argp.add_argument("-num", type = int)
    argp.add_argument("-round", type = int, default = 1, choices = [0, 1])
    args = argp.parse_args(sys.argv[1:7])
    
    model = args.model
    num = args.num
    is_round = bool(args.round)
    
    print("Start Training")
    function(model, num = num, is_round = is_round)