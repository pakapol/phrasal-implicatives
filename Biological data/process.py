import os
for f in os.listdir(os.getcwd()):
    print(f)
    if f[-8:] == "data.txt":
        write = []
        with open(f, "r") as f:
            new = []
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.strip() == "contradicts":
                    new += [lines[i+1], lines[i], lines[i-1], "\n"]
                if line.strip() == "entails":
                    new += [lines[i+1], "permits\n", lines[i-1], "\n"]
            write = lines + new
        with open("processed_" + f.name, "w") as f:
            f.writelines(write)
