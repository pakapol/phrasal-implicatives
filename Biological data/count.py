import os
temp = dict()
for q in os.listdir(os.getcwd()):
    if q[-8:] == "data.txt" and q[:3] == "pro":
        with open(q, "r") as f:
            lines = f.readlines()
            p = 0
            c = 0
            e = 0
            for line in lines:
                if "contradicts" in line:
                    c +=1
                if "permits" in line:
                    p +=1 
                if "entails" in line:
                    e += 1
            temp[q] = (e,c,p)
x = 0
for k in temp:
    print(k, "entails:", temp[k][0], "contradicts:", temp[k][1], "permits:",  temp[k][2])
    x += sum(temp[k])
print(x)
