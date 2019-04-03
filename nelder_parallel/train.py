import csv
import os
import numpy as np
from argparse import ArgumentParser as ArgPar

def load_var_features():
    with open("type_dict/type_dict.csv", "r", newline = "") as f:
        reader = csv.DictReader(f, delimiter = ";", quotechar = '"')
        
        var_names, type_dict, bounds, defaults, helps = [],[],[],[],[]

        for row in reader:
            var_names.append(row["var_name"])
            type_dict.append(eval(row["type"]))
            bounds.append(eval(row["bound"]))
            defaults.append(type_dict[-1](row["default"]))
            helps.append(row["help"])

        HyperParameters = get_arguments(var_names, type_dict, bounds, defaults, helps)

        return var_names, HyperParameters

def get_arguments(var_names, type_dict, bounds, defaults, helps):
    argp = ArgPar()
    n_vars = len(var_names) - 1

    for i in range(n_vars):
        argp.add_argument("-{}".format(var_names[i]), type = type_dict[i], default = defaults[i], help = helps[i])
    
    parsed_parameters = argp.parse_args()
    HyperParameters = {}

    for var_name, bound in zip(var_names[:-1], bounds[:-1]):
        HyperParameters[var_name] = np.clip(getattr(parsed_parameters, var_name), bound[0], bound[1])

    return HyperParameters


def renew_evaluation(var_names, x, y, num = 0):
    
    with open("evaluation/{}/evaluation.csv".format(num), "a", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = var_names, delimiter = ",", quotechar = '"')
        
        save_row = {var_name: value for var_name, value in zip(var_names, np.append(x, y))}
        writer.writerow(save_row)
        
class function():
    def __init__(self):
        self.var_names, self.HyperParameters = load_var_features()
        self.xs = np.array([self.HyperParameters["x1"], self.HyperParameters["x2"]])
        self.f()

    def f(self):
        y = np.sum(self.xs ** 2)
        print("Output: {:.3f}".format(y))
        print("")

        renew_evaluation(self.var_names, self.xs, y, num = 0)

if __name__ == "__main__":
    print("Start Training")
    function()