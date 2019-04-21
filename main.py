from argparse import ArgumentParser as ArgPar
import subprocess as sp
import sys
import math
import os
import time

class pycolor:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    END = '\033[0m'
    BOLD = '\038[1m'
    UNDERLINE = '\033[4m'
    INVISIBLE = '\033[08m'
    REVERCE = '\033[07m'    

if __name__ == "__main__":
    sp.call('export SINGULARITY_BINDPATH="~/research/test"', shell = True)

    try:
        argp = ArgPar()
        argp.add_argument("-model", type = str)
        argp.add_argument("-num", type = int)
        argp.add_argument("-itr", type = int, default = 60)
        argp.add_argument("-round", type = int, default = 1, choices = [0, 1])
        argp.add_argument("-cuda", type = int, nargs = "*", default = [1])
        argp.add_argument("-re", type = int, default = 0, choices = [0, 1])
        args = argp.parse_args()
        
        model = args.model
        num = args.num
        itr = args.itr
        round = bool(args.round)
        rerun = bool(args.re)
        cudas = args.cuda
        sys.argv[9]

    except Exception:
        print("") 
        print("###### ERROR ######")
        models = [file.split(".")[0] for file in os.listdir( "model" )]
        
        print("YOUR COMMAND MUST BE LIKE AS FOLLOWED:")
        print(pycolor.YELLOW + "python main.py -model CNN -num 0 -itr 10 -round 0 -cuda 0 1 -re 0" + pycolor.END)
        print("")
        print(pycolor.RED + "SET the variables shown below:" + pycolor.END)
        print("model: which model you run:")
        print("\t[", end = "")
        
        for model in models:
            if not "cache" in model:
                print("{}, ".format(model), end = "")
        print("]")
        print("")

        print("  num: how many times this experiment is: ")
        print("\t[0, 1, 2, ...]")
        print("")
        print("  itr: how many times evaluating the model: ")
        print("\t[any natural number]")
        print("")
        print("round: whether you round up the hyperparameter when one of them is out of boundaries.")
        print("\t1: Round up and Evaluate")
        print("\t0: Not Round up and Record -inf as output.")
        print("\t### NOTICE ### : When you examine some models by fixing the hyperparameters, try to set the round 1")
        print("")
        print(" cuda: Which cuda drivers do you want to make visible.")
        print("\t[any devices number's array. up to No.GPU you have in the device - 1]")
        print("\te.g. -cuda 0 1 # when you have at least 2 cuda drivers in your device.")
        print("")
        print("   re: When you want to restart the searching.")
        print("\t0 or 1 : Default is False(= 0)")
        print("\t1: Restart the searching")
        print("\t0: Set up from initilization.")
        print("")
        print("")
        sys.exit()
    
    if not os.path.isdir("shell/{}".format(model)):
        os.mkdir("shell/{}".format(model))
    if not os.path.isdir("shell/{}/{}".format(model, num)):
        os.mkdir("shell/{}/{}".format(model, num))

    if not os.path.isdir("evaluation/{}".format(model)):
        os.mkdir("evaluation/{}".format(model))
    if not os.path.isdir("evaluation/{}/{}".format(model, num)):
        os.mkdir("evaluation/{}/{}".format(model, num))

    if not os.path.isdir("log/{}".format(model)):
        os.mkdir("log/{}".format(model))
    if not os.path.isdir("log/{}/{}".format(model, num)):
        os.mkdir("log/{}/{}".format(model, num))

    if not os.path.isdir("exec_screen/{}".format(model)):
        os.mkdir("exec_screen/{}".format(model))
    if not os.path.isdir("exec_screen/{}/{}".format(model, num)):
        os.mkdir("exec_screen/{}/{}".format(model, num))

    if not os.path.isdir("operation/{}".format(model)):
        os.mkdir("operation/{}".format(model))
    if not os.path.isdir("operation/{}/{}".format(model, num)):
        os.mkdir("operation/{}/{}".format(model, num))

    if not os.path.isdir("simplex/{}".format(model)):
        os.mkdir("simplex/{}".format(model))
    if not os.path.isdir("simplex/{}/{}".format(model, num)):
        os.mkdir("simplex/{}/{}".format(model, num))

    if not os.path.isdir("storage/{}".format(model)):
        os.mkdir("storage/{}".format(model))
    if not os.path.isdir("storage/{}/{}".format(model, num)):
        os.mkdir("storage/{}/{}".format(model, num))    

    init_sh = \
        ["#!/bin/bash", \
        "USER=$(whoami)", \
        "\n", \
        "rm storage/{}/{}/storage.csv".format(model, num), \
        "echo $USER:~$PWD$ rm storage/{}/{}/storage.csv".format(model, num), \
        "rm simplex/{}/{}/simplex.csv".format(model, num), \
        "echo $USER:~$PWD$ rm simplex/{}/{}/simplex.csv".format(model, num), \
        "rm operation/{}/{}/operations.csv".format(model, num), \
        "echo $USER:~$PWD$ rm operation/{}/{}/operations.csv".format(model, num), \
        "rm evaluation/{}/{}/evaluation.csv".format(model, num), \
        "echo $USER:~$PWD$ rm evaluation/{}/{}/evaluation.csv".format(model, num), \
        "rm log/{}/{}/*.csv".format(model, num), \
        "echo $USER:~$PWD$ rm log/{}/{}/*.csv".format(model,num), \
        "rm exec_screen/{}/{}/*.log".format(model, num), \
        "echo $USER:~$PWD$ rm exec_screen/{}/{}/*.log".format(model,num), \
        ]

    main_sh = \
        ["#!/bin/bash", \
        "USER=$(whoami)", \
        "\n", \
        "echo $USER:~$PWD$ python nelder.py -model {} -num {} -round {}".format(model, num, int(round)), \
        "python nelder.py -model {} -num {} -round {}".format(model, num, int(round)), \
        "echo",  \
        "echo $USER:~$PWD$ python env.py -model {} -num {} -round {} -cuda {}".format(model, num, int(round), 0), \
        "python env.py -model {} -num {} -round {} -cuda {}".format(model, num, int(round), 0), \
        "echo",  \
        "echo $USER:~$PWD$ ./shell/{}/{}/run.sh".format(model, num), \
        "chmod +x shell/{}/{}/run.sh".format(model, num), \
        "./shell/{}/{}/run.sh".format(model, num), \
        "echo",  \
        ]
    
    cuda_script_idx = [i for i, s in enumerate(main_sh) if "-cuda" in s]
    n_cudas = len(cudas)
    script = ""

    if not rerun:
        files = [
                    ["operation/{}/{}/".format(model, num) + f for f in os.listdir("operation/{}/{}".format(model, num))],\
                    ["simplex/{}/{}/".format(model, num) + f for f in os.listdir("simplex/{}/{}".format(model, num))],\
                    ["storage/{}/{}/".format(model, num) + f for f in os.listdir("storage/{}/{}".format(model, num))],\
                    ["log/{}/{}/".format(model, num) + f for f in os.listdir("log/{}/{}".format(model, num))],\
                    ["exec_screen/{}/{}/".format(model, num) + f for f in os.listdir("exec_screen/{}/{}".format(model, num))],\
                    ["evaluation/{}/{}/".format(model, num) + f for f in os.listdir("evaluation/{}/{}".format(model, num))] ]
        
        rm_files = []

        for line in init_sh:
            script += line + "\n"
        
        for fs in files:
            for file in fs:
                rm_files.append(file)

        if len(rm_files) > 0:
            print("")
            print("#### NOTICE ###") 
            print(pycolor.YELLOW + "Model is {} and the trial number is {}".format(model, num) + pycolor.END)
            print(pycolor.RED + "There are some files you are going to remove." + pycolor.END)
            print("")
            
            for i, file in enumerate(rm_files):
                if i % 3 == 0 and i != 0:
                    print("")    
                print(pycolor.GREEN + "{0:<45}".format(file) + pycolor.END, end = "")

            
            answer = ""
            while not answer in {"y", "n"}:
                print("")
                print("")
                answer = input("Is it okay? [y or n] : ")
        
            if answer == "y":
                with open("shell/{}/{}/init.sh".format(model, num), "w") as f:
                    f.writelines(script)
            else:
                print("Permission Denied.")
                sys.exit()

            print("")
            print("#########################")
            print("# WILL REMOVE THE FILES #")
            print("#########################")
            print("")

            sp.call("chmod +x shell/{}/{}/init.sh".format(model, num), shell = True)
            sp.call("./shell/{}/{}/init.sh".format(model, num), shell = True)
            
            print("")
            print("#########################")
            print("### REMOVED THE FILES ###")
            print("#########################")
            print("")
    else:
        n_log = len(os.listdir("log/{}/{}".format(model, num)))
        n_ex = len(os.listdir("exec_screen/{}/{}".format(model, num)))
        for del_idx in range(n_log, n_ex):
            sp.call("rm {}".format("exec_screen/{}/{}/exec{}.log".format(model, num, del_idx)), shell = True)
       
        with open("shell/{}/{}/init.sh".format(model, num), "w") as f:
            f.writelines(script)

    print("")
    print("#########################")
    print("##### RENEW THE ENV #####")
    print("#########################")
    print("")

        
    if not os.path.isfile("storage/{}/{}/storage.csv".format(model, num)):
        with open("storage/{}/{}/storage.csv".format(model, num), "w", newline = "") as f:
            pass
        
    if not os.path.isfile("simplex/{}/{}/simplex.csv".format(model, num)):
        with open("simplex/{}/{}/simplex.csv".format(model, num), "w", newline = "") as f:
            pass

    if not os.path.isfile("operation/{}/{}/operations.csv".format(model, num)):
        with open("operation/{}/{}/operations.csv".format(model, num), "w", newline = "") as f:
            pass

    for t in range(itr):
        n_log = len(os.listdir("exec_screen/{}/{}".format(model, num)))
        print("")
        print("")
        print("#########################")
        print("####### evals {:0>3} #######".format(t + 1 + n_log))
        print("#########################")
        print("")
        print("Will use the cuda visible device {}".format(cudas[(t + 1) % n_cudas]))
        print("")
        print("")
        script = ""
        
        for idx in cuda_script_idx:
            main_sh[idx] = main_sh[idx].split("-cuda")[0] + "-cuda {}".format(cudas[(t + 1) % n_cudas])

        for line in main_sh:
            script += line + "\n"

        with open("shell/{}/{}/main.sh".format(model, num), "w") as f:
            f.writelines(script)
        
        sp.call("chmod +x shell/{}/{}/main.sh".format(model, num), shell = True)
        sp.call("./shell/{0}/{1}/main.sh > exec_screen/{0}/{1}/exec{2}.log".format(model, num, n_log), shell = True)
