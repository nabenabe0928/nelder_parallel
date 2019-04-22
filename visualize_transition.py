import csv
import matplotlib.pyplot as plt
import os
import sys
from argparse import ArgumentParser

def get_evaluations(model, num):
    with open("evaluation/{}/{}/evaluation.csv".format(model, num), "r", newline = "") as f:
        reader = csv.reader(f, delimiter = ",", quotechar = '"')
        evaluations = []

        for row in reader:
            if "." in row[-1] or "0" in row[-1]:
                evaluations.append(float(row[-1]))
    
    return evaluations

def get_accuracy(model, num):
    n_log = len(os.listdir("log/{}/{}".format(model, num)))
    name = "TestAcc"
    accs = []

    for n in range(n_log):
        
        with open("log/{}/{}/eval{}.csv".format(model, num, n), "r", newline = "") as f:
            reader = csv.DictReader(f, delimiter = "\t", quotechar = '"')
            max_acc = 0
            
            for row in reader:
                v = float(row[name])
                if v > max_acc:
                    max_acc = v

            accs.append(max_acc)

    return accs

def include_barrier_loss(evals, model, num):
    xs, ys = [], []
    min_max = 0
    for x, y in enumerate(evals):
        xs.append(x)
        ys.append(y)
        if min_max < y < 1.0e+7:
            min_max = y
    
    plt.figure()
    plt.ylim(0, min_max)
    plt.title("transition including barrier function")
    plt.xlabel("The number of evaluations")
    plt.ylabel("Test Loss")
    plt.grid()
    plt.plot(xs, ys)
    plt.savefig("figures/{}/{}/in_barrier_loss.png".format(model, num))

def exclude_barrier_loss(evals, model, num):
    xs, ys = [], []
    min_max = 0
    for x, y in enumerate(evals):
        if y < 1.0e+7:
            xs.append(x)
            ys.append(y)
            if min_max < y:
                min_max = y
    
    plt.figure()
    plt.ylim(0, min_max)
    plt.title("transition excluding barrier function")
    plt.grid()
    plt.xlabel("The number of evaluations")
    plt.ylabel("Test Loss")
    plt.plot(xs, ys)
    plt.savefig("figures/{}/{}/ex_barrier_loss.png".format(model, num))

def bottom_line_loss(evals, model, num):
    xs, ys = [], []
    min_max = 0
    for x, y in enumerate(evals):
        if y < 1.0e+7:
            xs.append(x)
            
            if len(ys) > 0:
                ys.append(min(y, ys[-1]))
            else:
                ys.append(y)
                
            if min_max < y:
                min_max = y
    
    plt.figure()
    plt.ylim(0, min_max)
    plt.title("transition of bottom line")
    plt.xlabel("The number of evaluations")
    plt.ylabel("Test Loss")
    plt.grid()
    plt.plot(xs, ys)
    plt.savefig("figures/{}/{}/bottom_loss.png".format(model, num))

def include_barrier_acc(evals, model, num):
    xs, ys = [], []
    max_acc = 0.
    for x, y in enumerate(evals):
        xs.append(x)
        ys.append(y)
        if max_acc < y:
            max_acc = y
    
    plt.figure()
    plt.ylim(0, max_acc)
    plt.title("transition of accuracy including barrier function")
    plt.xlabel("The number of evaluations")
    plt.ylabel("Test Accuracy")
    plt.grid()
    plt.plot(xs, ys)
    plt.savefig("figures/{}/{}/in_barrier_acc.png".format(model, num))

def exclude_barrier_acc(evals, model, num):
    xs, ys = [], []
    max_acc = 0
    for x, y in enumerate(evals):
        if y != 0:
            xs.append(x)
            ys.append(y)
            if max_acc < y:
                max_acc = y
    
    plt.figure()
    plt.ylim(0, max_acc)
    plt.title("transition of accuracy excluding barrier function")
    plt.xlabel("The number of evaluations")
    plt.ylabel("Test Accuracy")
    plt.grid()
    plt.plot(xs, ys)
    plt.savefig("figures/{}/{}/ex_barrier_acc.png".format(model, num))

def top_line_acc(evals, model, num):
    xs, ys = [], []
    max_acc = 0
    for x, y in enumerate(evals):
        if y != 0:
            xs.append(x)
            
            if len(ys) > 0:
                ys.append(max(y, ys[-1]))
            else:
                ys.append(y)
                
            if max_acc < y:
                max_acc = y
    
    plt.figure()
    plt.ylim(0, max_acc)
    plt.title("transition of top line")
    plt.xlabel("The number of evaluations")
    plt.ylabel("Test Accuracy")
    plt.grid()
    plt.plot(xs, ys)
    plt.savefig("figures/{}/{}/top_acc.png".format(model, num))

if __name__== "__main__":
    models = os.listdir("evaluation")
    argp = ArgumentParser()
    argp.add_argument("-model", choices = models)
    argp.add_argument("-num")
    args = argp.parse_args()
    model = args.model
    num = args.num

    if model == None or num == None:
        print("-model and -num must be set.")
        
        model_choice = ""
        for m in models:
            model_choice += m + ", "
        print("-model choice is ({})".format(model_choice))
        print("-num is any arbitrary integer included in evaluation/[model].")
        sys.exit()
    
    if not os.path.isdir("figures"):
        os.mkdir("figures")
    if not os.path.isdir("figures/{}".format(model)):
        os.mkdir("figures/{}".format(model))
    if not os.path.isdir("figures/{}/{}".format(model, num)):
        os.mkdir("figures/{}/{}".format(model, num))    

    evals = get_evaluations(model, num)
    accs = get_accuracy(model, num)
    
    include_barrier_loss(evals, model, num)
    exclude_barrier_loss(evals, model, num)
    bottom_line_loss(evals, model, num)
    
    include_barrier_acc(accs, model, num)
    exclude_barrier_acc(accs, model, num)
    top_line_acc(accs, model, num)