import os
temp = dict()
for q in os.listdir(os.getcwd()):
    if q[-8:] == "data.txt" and q[:3] == "pro":
        with open(q, "r") as f:
            lines = f.readlines()
            temp[q] = len(lines)/4
x = 0
for k in temp:
    print(k, temp[k])
    x += temp[k]
print(x)

