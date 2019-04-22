import csv
import matplotlib.pyplot as plt
import os
import sys
from argparse import ArgumentParser

def get_operations(model, num):
    with open("operation/{}/{}/operations.csv".format(model, num), "r", newline = "") as f:
        reader = csv.reader(f, delimiter = ",", quotechar = '"')
        operations = []

        for row in reader:
            operations.append(row[-1])
    
    return operations

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

    counter = {"i": 0, "s": 0, "r": 0, "ic": 0, "oc": 0, "e": 0}
    ops = get_operations(model, num)
    all = 0

    for op in ops:
        counter[op] += 1
        all += 1
    
    percents = {k : v / all * 100. for k, v in counter.items()}
    print(percents)

