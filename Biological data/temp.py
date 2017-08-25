import os

for q in os.listdir(os.getcwd()):
    if q[-8:] == "data.txt" and q[:2] == "n_":
        with open(q[2:], "r") as f:
            lines1 = f.readlines()
        with open(q, "r") as g:
            lines2 = g.readlines()
        write = lines1 + lines2
        with open(q[2:], "w") as k:
            k.writelines(write)

