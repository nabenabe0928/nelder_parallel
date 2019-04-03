import csv

def load_storage():
    with open("storage/storage.csv", "r", newline = "") as f:
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
        with open("storage/storage.csv", "w", newline = "") as f:
            writer = csv.DictWriter(f, fieldnames = var_names, delimiter = ",", quotechar = '"')
            writer.writeheader()
    else:
        renew_storage(post_storage)
    
    generate_shell(target)

def renew_storage(post_storage):
    var_names = [var_name for var_name in post_storage[0].keys()]

    with open("storage/storage.csv", "w", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = var_names, delimiter = ",", quotechar = '"')
        writer.writeheader()

        save_row = {}

        for x in post_storage:
            for var_name, value in x.items():
                save_row[var_name] = value
            writer.writerow(save_row)

def generate_shell(target):
    scripts = ["#!/usr/bin/bash","USER=$(whoami)","CWD=dirname $0", ]
    enter = "\n"
    first_script = ""
    second_script = ""

    for name in target.keys():
        print("    {}\t".format(name), end = "")
    print("")
    for value in target.values():
        print("  {:.2f}\t".format(float(value)), end = "")    
    print("")
    print("")

    for s in scripts:
        first_script += s + enter
    
    second_script = "python train.py "
    
    for var_name, value in target.items():
        second_script += "-{} {} ".format(var_name, value)

    script = first_script + "echo $USER:~$CWD$ {} \n".format(second_script) + second_script

    with open("run.sh", "w") as f:
        f.writelines(script)

def main():
    load_storage()

if __name__ == "__main__":
    print("Collecting Environment Variables and Putting in Shell Scripts.")
    main()