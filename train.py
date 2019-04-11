import csv
import os
import numpy as np
from argparse import ArgumentParser as ArgPar
from collections import namedtuple
import sys
import math
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import datetime
from torch.autograd import Variable
from dataset import get_data
from tqdm import tqdm
from model.WideResNet import WideResNet
from model.CNN import CNN

def renew_storage(model, num, is_round):
    with open("storage/{}/{}/storage.csv".format(model, num), "r", newline = "") as f:
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
        with open("storage/{}/{}/storage.csv".format(model, num), "w", newline = "") as f:
            writer = csv.DictWriter(f, fieldnames = var_names, delimiter = ",", quotechar = '"')
            writer.writeheader()
    else:
        del_storage(post_storage, model, num)
            
def del_storage(post_storage, model, num):
    var_names = [var_name for var_name in post_storage[0].keys()]

    with open("storage/{}/{}/storage.csv".format(model, num), "w", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = var_names, delimiter = ",", quotechar = '"')
        writer.writeheader()

        save_row = {}

        for x in post_storage:
            for var_name, value in x.items():
                save_row[var_name] = value
            writer.writerow(save_row)

def load_var_features(model, is_round = True):
    with open("type_dict/{}/type_dict.csv".format(model), "r", newline = "") as f:
        reader = csv.DictReader(f, delimiter = ";", quotechar = '"')
        
        var_names, type_dict, bounds, defaults, dists = [],[],[],[],[]

        for row in reader:
            var_names.append(row["var_name"])
            type_dict.append(eval(row["type"]))
            bounds.append(eval(row["bound"]))
            defaults.append(type_dict[-1](eval(row["default"])))
            dists.append(row["dist"])

        HyperParameters, out_of_bound = get_arguments(var_names, type_dict, bounds, defaults, is_round)

        return var_names, HyperParameters, out_of_bound, dists

def convert_value_by_dist(value, dist):
    if "log" in dist:
        base = float(dist.split("log")[-1])
        v = math.log(value, base)
    else:
        v = value
    return v

def re_storage(hp_dict, dists, model, num):
    
    with open("storage/{}/{}/storage.csv".format(model, num), "a", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = hp_dict.keys(), delimiter = ",", quotechar = '"')

        save_row = {var_name: convert_value_by_dist(x, dist) for (var_name, x), dist in zip(hp_dict.items(), dists)}
        writer.writerow(save_row)

def get_arguments(var_names, type_dict, bounds, defaults, is_round = True):
    argp = ArgPar()
    n_vars = len(var_names) - 1

    for i in range(n_vars):
        argp.add_argument("-{}".format(var_names[i]), type = type_dict[i], default = defaults[i])
    
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
        
        if type(HyperParameters[var_name]) == np.int64:
            HyperParameters[var_name] = int(HyperParameters[var_name])
        
    return HyperParameters, out_of_bound

def renew_evaluation(y_name, hp, y, model, num = 0):
    hp[y_name] = y
    with open("evaluation/{}/{}/evaluation.csv".format(model, num), "a", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = hp.keys(), delimiter = ",", quotechar = '"')
        
        writer.writerow(hp)

def accuracy(y, target):
        pred = y.data.max(1, keepdim = True)[1]
        acc = pred.eq(target.data.view_as(pred)).cpu().sum()
        return acc

def record_log(rsl, model, num):
    if not os.path.isdir("log"):
        os.mkdir("log")
    if not os.path.isdir("log/{}".format(model)):
        os.mkdir("log/{}".format(model))
    if not os.path.isdir("log/{}/{}".format(model, num)):
        os.mkdir("log/{}/{}".format(model, num))
    
    n_evals = len( os.listdir(os.getcwd() + "/log/{}/{}".format(model, num) ) )

    with open("log/{}/{}/eval{}.csv".format(model, num, n_evals + 1), "w", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = rsl[0].keys(), delimiter = "\t", quotechar = "'")
        writer.writeheader()

        for row in rsl:
            writer.writerow(row)

def print_result(values):
    f1 = "  {"
    f2 = "}  "
    f_int = "{}:<20"
    f_float = "{}:<20.5f"

    f_vars = ""
    
    for i, v in enumerate(values):
        if type(v) == float:
            f_vars += f1 + f_float.format(i) + f2
        else:
            f_vars += f1 + f_int.format(i) + f2
    
    print(f_vars.format(*values))
            
def train(learner, model_name, num):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data, test_data = get_data(learner.batch_size)
    
    learner = learner.to(device)
    cudnn.benchmark = True

    optimizer = optim.SGD( \
                        learner.parameters(), \
                        lr = learner.lr, \
                        momentum = learner.momentum, \
                        weight_decay = learner.weight_decay, \
                        nesterov = True \
                        )
    
    loss_func = nn.CrossEntropyLoss().cuda()

    milestones = learner.lr_step
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = learner.lr_decay)


    rsl_keys = ["lr", "epoch", "TrainAcc", "TrainLoss", "TestAcc", "TestLoss", "Time"]
    rsl = []
    y_out = 1.0e+8
    
    print_result(rsl_keys)
    
    for epoch in range(learner.epochs):
        lr = optimizer.param_groups[0]["lr"]
        learner.train()
        bar = tqdm(desc = "Training", total = len(train_data), leave = False)
        train_loss, train_acc, n_train, test_loss, test_acc, n_test = 0, 0, 0, 0, 0, 0

        for data, target in train_data:
            data, target =  data.to(device), target.to(device) 
            y = learner(data)
            loss = loss_func(y, target)
            optimizer.zero_grad() # clears the gradients of all optimized tensors.
            loss.backward() 
            optimizer.step() # renew learning rate

            train_acc += accuracy(y, target)

            train_loss += loss.item() * target.size(0)
            n_train += target.size(0)

            bar.set_description("Loss: {0:.6f}, Accuracy: {1:.6f}".format(train_loss / n_train, float(train_acc) / n_train))
            bar.update()

        bar.close()
        bar = tqdm(desc = "Testing", total = len(test_data), leave = False)
        learner.eval() # switch to test mode( make model not save the record of calculation)

        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(device), target.to(device)
                y = learner(data)
                loss = loss_func(y, target)

                test_acc += accuracy(y, target)
                test_loss += loss.item() * target.size(0)
                n_test += target.size(0)

                bar.update()
        bar.close()

        train_loss, train_acc = train_loss / n_train, float(train_acc) / n_train
        test_loss, test_acc = test_loss / n_test, float(test_acc) / n_test

        time_now = str(datetime.datetime.today())
        rsl.append({k: v for k, v in zip(rsl_keys, [lr, epoch + 1, train_acc, train_loss, test_acc, test_loss, time_now])})
        
        y_out = min(y_out, test_loss)
        
        print_result(rsl[-1].values())

        scheduler.step()

    record_log(rsl, model_name, num)

    return y_out

class function():
    def __init__(self, model, num = 0, is_round = True):
        self.num = num
        self.model = model
        self.is_round = is_round
        self.var_names, HyperParameters, self.out_of_bound, self.dists = load_var_features(self.model, is_round = self.is_round)
        self.hp_dict = HyperParameters

        hp_tuple = namedtuple("_hyperparameters", (var_name for var_name in self.var_names[:-1]))
        self.hyperparameters = hp_tuple(**self.hp_dict)
        
        try:
            self.f()
        except Exception as exc:
            print("FAILED to EVALUATE THE HYPERPARAMETERS, BECAUSE OF ERROR BELOW")
            print(exc)
            print("TRY to EVALUATE AGAIN")
            re_storage(self.hp_dict, self.dists, model, num)

    def f(self):
        if not self.out_of_bound:
            learner = eval(self.model)(self.hyperparameters)
            loss = train(learner, self.model, self.num)
            print("Output: {:.3f}".format(loss))
            print("")
        else:
            loss = 1.0e+08
            print("### The value was out of boundary. ###")
            print("Output: {:.3f}".format(loss))
            print("")

        renew_evaluation(self.var_names[-1], self.hp_dict, loss, self.model, num = self.num)
        renew_storage(self.model, self.num, self.is_round)

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
    print("")
    print("")
    function(model, num = num, is_round = is_round)