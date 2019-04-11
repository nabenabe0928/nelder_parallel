import csv
import numpy as np
import random
import copy
import math
import os
from argparse import ArgumentParser as ArgPar

def renew_simplex(xs, ys, var_names,type_dict, model, num):
    with open("simplex/{}/{}/simplex.csv".format(model, num), "w", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = var_names, delimiter = ",", quotechar = '"')
        writer.writeheader()

        save_row = {}
        
        for x in np.c_[xs, ys]:
            for var_name, value, vtype in zip(var_names, x, type_dict):
                save_row[var_name] = vtype(value)
            writer.writerow(save_row)


def renew_operations(ops, model, num = 0):
    with open("operation/{}/{}/operations.csv".format(model, num), "a", newline = "") as f:
        writer = csv.writer(f, delimiter = ",", quotechar = '"')

        for op in ops:
            writer.writerow([op])

def load_operations(model, num = 0):
    
    with open("operation/{}/{}/operations.csv".format(model, num), "r", newline = "") as f:
        reader = csv.reader(f, delimiter = ",", quotechar = '"')
        
        ops = []
        
        for row in reader:
            ops.append(row[0])

        return ops

def load_storage(model, num):
    with open("storage/{}/{}/storage.csv".format(model, num), "r", newline = "") as f:
        reader = csv.DictReader(f, delimiter = ",", quotechar = '"')
        cnt = 0

        for _ in reader:
            cnt += 1

        return cnt


def print_result(values):
    f1 = "  {"
    f2 = "}  "
    f_int = "{}:<13"
    f_float = "{}:<13.3f"

    f_vars = ""
    
    for i, v in enumerate(values):
        if type(v) == float:
            f_vars += f1 + f_float.format(i) + f2
        else:
            f_vars += f1 + f_int.format(i) + f2
    
    print(f_vars.format(*values))

def renew_storage(storage, var_names, model, num):
    with open("storage/{}/{}/storage.csv".format(model, num), "w", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = var_names, delimiter = ",", quotechar = '"')
        writer.writeheader()

        save_row = {}

        print("Saving the position data as shown below...")
        print_result(var_names)
        
        for x in storage:
            print_result(x[:-1])
            for var_name, value in zip(var_names, x):
                save_row[var_name] = value
            
            writer.writerow(save_row)
        print("")

def load_simplex(var_names, type_dict, n_points, model, num):
    
    with open("simplex/{}/{}/simplex.csv".format(model, num), "r", newline = "") as f:
        reader = csv.DictReader(f, delimiter = ",", quotechar = '"')
        
        n_vars = len(var_names[:-1])

        xs = np.array([[None for _ in range(n_vars)] for __ in range(n_points)])
        ys = np.array([None for _ in range(n_points)])

        for point_i, row in enumerate(reader):
            for var_i, var_name in enumerate(var_names[:-1]):
                xs[point_i][var_i] = type_dict[var_i](row[var_name])
            
            ys[point_i] = type_dict[-1](row[ var_names[-1] ])
        
        return xs, ys

def convert_attr_by_dist(vtype, bound, dist):
    if "log" in dist:
        base = float(dist.split("log")[-1])
        bound = [ math.log(bound[0], base), math.log(bound[1], base) ]
        vtype = float
    return vtype, bound

def load_var_features(model):
    with open("type_dict/{}/type_dict.csv".format(model), "r", newline = "") as f:
        reader = csv.DictReader(f, delimiter = ";", quotechar = '"')
        
        var_names, type_dict, bounds, dists = [],[],[],[]

        for row in reader:
            vtype, bound = convert_attr_by_dist(eval(row["type"]), eval(row["bound"]), row["dist"]) 
            
            var_names.append(row["var_name"])
            type_dict.append(vtype)
            bounds.append(bound)
            dists.append(row["dist"])

        return var_names, type_dict, bounds, dists

def load_evaluation(var_names, type_dict, dists, model, num = 0):   
    if not os.path.isfile("evaluation/{}/{}/evaluation.csv".format(model, num)):
        with open("evaluation/{}/{}/evaluation.csv".format(model, num), "w") as f:
            writer = csv.DictWriter(f, fieldnames = var_names, delimiter = ",", quotechar = '"')
            writer.writeheader()

    with open("evaluation/{}/{}/evaluation.csv".format(model, num), "r", newline = "") as f:
        reader = csv.DictReader(f, delimiter = ",", quotechar = '"')
        prior_xs = []
        prior_ys = []

        for row in reader:
            prior_xs.append([])
            for var_name, vtype, dist in zip(var_names[:-1], type_dict[:-1], dists[:-1]):
                v = convert_value_by_dist(row[var_name], dist, vtype)
                prior_xs[-1].append(v)
            prior_ys.append( type_dict[-1](row[var_names[-1]]) )
    
    return np.array(prior_xs), np.array(prior_ys)

def convert_value_by_dist(value, dist, vtype):
    if "log" in dist:
        base = float(dist.split("log")[-1])
        v = vtype(math.log(float(value), base))
    else:
        v = vtype(value)
    return v

def latin_hypercube_sampling(bounds, n_points, type_dict):
    n_dim = len(bounds)
    rnd_grid = np.array([np.random.permutation(list(range(1, n_points + 1))) for _ in range(n_dim)])
    
    xs = [[bi[0] + (rnd_grid[i][s] - random.random()) * (bi[1] - bi[0]) / n_points for i, bi in enumerate(bounds)
        ] for s in range(n_points)]
    
    return xs

class nelder():
    
    def __init__(self, model, num = 0, is_round = True):
        self.num = num
        self.model = model
        self.is_round = is_round

        self.var_names, self.type_dict, self.bounds, self.dists = load_var_features(self.model)
        self.n_points = self.get_dim()
        self.xs, self.ys = load_simplex(self.var_names, self.type_dict, self.n_points, self.model, self.num)
        self.prior_xs, self.prior_ys = load_evaluation(self.var_names, self.type_dict, self.dists, self.model, num = self.num)

        self.operations = load_operations(self.model, num = self.num)
        self.storage = []
        self.initilize = False

        if len(self.operations) == 0:
            self.xs = latin_hypercube_sampling(self.bounds, self.n_points, self.type_dict)
            self.ys = np.array([1.0e+08 for _ in range(self.n_points)])
            self.initilize = True
                    
        self.c = None
        self.coef = {"r": 1.0, "e": 2.0, "ic": - 0.5, "oc": 0.5, "s": 0.5}
        
        if self.initilize:
            last_op = None
        else:    
            last_op = self.operations[-1]

        self.main(last_op)

    def get_dim(self):
        n_dim = 1
        
        for bound in self.bounds[:-1]:
            if not bound[0] == bound[1]:
                n_dim += 1
        
        return n_dim

    def order_by(self):
        order = np.argsort(self.ys)

        self.xs = self.xs[order]
        self.ys = self.ys[order]

    def centroid(self):
        self.xs = np.array(self.xs)
        self.ys = np.array(self.ys)
        self.order_by()
        self.c = self.xs[:-1].mean(axis = 0)

    def reflect(self):
        self.centroid()
        xr = self.c + self.coef["r"] * (self.c - self.xs[-1])
        
        if self.is_round:
            for i, (xi, bound) in enumerate(zip(xr, self.bounds)):
                xr[i] = np.clip(xi, bound[0], bound[1])
        

        return xr
        
    def expand(self):
        self.centroid()
        xe = self.c + self.coef["e"] * (self.c - self.xs[-1])

        if self.is_round:
            for i, (xi, bound) in enumerate(zip(xe, self.bounds)):
                xe[i] = np.clip(xi, bound[0], bound[1])
        
        return xe

    def outside_contract(self):
        self.centroid()
        xoc = self.c + self.coef["oc"] * (self.c - self.xs[-1])

        if self.is_round:
            for i, (xi, bound) in enumerate(zip(xoc, self.bounds)):
                xoc[i] = np.clip(xi, bound[0], bound[1])
        
        return xoc

    def inside_contract(self):
        self.centroid()
        xic = self.c + self.coef["ic"] * (self.c - self.xs[-1])
        
        return xic

    def shrink(self):
        self.centroid()
        for i in range(1, self.n_points):
            self.xs[i] = self.xs[0] + self.coef["s"] * (self.xs[i] - self.xs[0])
                        
    def main(self, last_ops):
        next_ops = []

        if self.initilize:
            self.storage = [x[:] for x in self.xs]
            next_ops = ["i" for _ in range(self.n_points)]
            
        elif last_ops == "i":
            xis = self.prior_xs[- self.n_points:]
            yis = self.prior_ys[- self.n_points:]
            
            for i in range(self.n_points):
                self.xs[i] = copy.deepcopy(xis[i])
                self.ys[i] = yis[i]
            
            xr = self.reflect()
            self.storage = [xr[:]]
            next_ops.append("r")
        
        elif last_ops == "s":
            xss = self.prior_xs[- self.n_points + 1:]
            yss = self.prior_ys[- self.n_points + 1:]

            for i in range(self.n_points - 1):
                self.xs[i + 1] = copy.deepcopy(xss[i])
                self.ys[i + 1] = yss[i]
            
            xr = self.reflect()
            self.storage = [xr[:]]
            next_ops.append("r")

        elif last_ops == "r":
            xr = self.prior_xs[-1]
            yr = self.prior_ys[-1]
            
            if self.ys[0] <= yr < self.ys[-2]:
                self.xs[-1] = xr[:]
                self.ys[-1] = yr
                xrr = self.reflect()
                self.storage = [xrr[:]]
                next_ops.append("r")
            
            elif yr < self.ys[0]:
                xe = self.expand()
                self.storage = [xe[:]]
                next_ops.append("e")

            elif self.ys[-2] <= yr < self.ys[-1]:
                xoc = self.outside_contract()
                self.storage = [xoc[:]]
                next_ops.append("oc")

            elif self.ys[-1] <= yr:
                xic = self.inside_contract()
                self.storage = [xic[:]]
                next_ops.append("ic")

        elif last_ops == "ic":
            xic = self.prior_xs[-1]
            yic = self.prior_ys[-1]
            
            if yic < self.ys[-1]:
                self.xs[-1] = xic[:]
                self.ys[-1] = yic
                xicr = self.reflect()
                self.storage = [xicr[:]]
                next_ops.append("r")

            else:
                self.shrink()
                self.storage = [x[:] for x in self.xs[1:]]
                next_ops = ["s" for _ in range(self.n_points - 1)]
        
        elif last_ops == "oc":
            xoc = self.prior_xs[-1]
            yoc = self.prior_ys[-1]
            yr = self.prior_ys[-2]

            if yoc <= yr:
                self.xs[-1] = xoc[:]
                self.ys[-1] = yoc
                xocr = self.reflect()
                self.storage = [xocr[:]]
                next_ops.append("r")
            
            else:
                self.shrink()
                self.storage = [x[:] for x in self.xs[1:]]
                next_ops = ["s" for _ in range(self.n_points - 1)]
        
        elif last_ops == "e":
            xe = self.prior_xs[-1]
            ye = self.prior_ys[-1]
            xr = self.prior_xs[-2]
            yr = self.prior_ys[-2]

            if ye < yr:
                self.xs[-1] = xe[:]
                self.ys[-1] = ye
                xer = self.reflect()
                self.storage = [xer[:]]
                next_ops.append("r")

            else:
                self.xs[-1] = xr[:]
                self.ys[-1] = yr
                xrr = self.reflect()
                self.storage = [xrr[:]]
                next_ops.append("r")

        storage = [[t(s) for s, t in zip(st, self.type_dict)] for st in self.storage]
        xs = [[t(xi) for xi, t in zip(x, self.type_dict)] for x in self.xs]

        renew_storage(storage, self.var_names[:-1], self.model, self.num)
        renew_simplex(xs, self.ys, self.var_names, self.type_dict, self.model, self.num)
        renew_operations(next_ops, self.model, num = self.num)

if __name__ == "__main__":
    argp = ArgPar()
    argp.add_argument("-model", type = str)
    argp.add_argument("-num", type = int)
    argp.add_argument("-round", type = int, default = 1, choices = [0, 1])
    args = argp.parse_args()
    
    model = args.model
    num = args.num
    is_round = bool(args.round)

    if load_storage(model, num) == 0:
        print("Will get the positions by Nelder-Mead Method.")
        nelder(model, num = num, is_round = is_round)
    else:
        print("Will collect a position from the storage.")
        print("")