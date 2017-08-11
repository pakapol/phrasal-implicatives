import sys
import string
import os

def main():
    with open("embedded-constructions.txt", "r") as f:
        constrs = [ line.strip() for line in f]
    counts = dict()
    output = []
    for c in constrs:
        counts[c] = 0
    for dir in os.listdir(os.path.join(os.getcwd(), "disc1/gigaword_eng_5_d1/data")):
        for filename in os.listdir(os.path.join(os.path.join(os.getcwd(), "disc1/gigaword_eng_5_d1/data"), dir)):
            with open(os.path.join(os.path.join(os.path.join(os.getcwd(), "disc1/gigaword_eng_5_d1/data"), dir),filename), "r", encoding="utf8") as f:
                content = f.read()
                for c in constrs:
                    k=0
                    while k != -1:
                        k =content.find(c, k+1)
                        if k != -1:
                            output.append(content[content.rfind("<P>", 0,k)+4:content.find("</P>", k)] + "\n")
                            print(output[-1])
                            counts[c] += content.count(c)
    for dir in os.listdir(os.path.join(os.getcwd(), "disc2/gigaword_eng_5_d2/data")):
        for filename in os.listdir(os.path.join(os.path.join(os.getcwd(), "disc2/gigaword_eng_5_d2/data"), dir)):
            with open(os.path.join(os.path.join(os.path.join(os.getcwd(), "disc2/gigaword_eng_5_d2/data"), dir),filename), "r", encoding="utf8") as f:
                content = f.read()
                for c in constrs:
                    k=0
                    while k != -1:
                        k = content.find(c, k+1)
                        if k != -1:
                            output.append(content[content.rfind("<P>", 0,k)+4:content.find("</P>", k)] + "\n")
                            print(output[-1])
                            counts[c] += content.count(c)
    for dir in os.listdir(os.path.join(os.getcwd(), "disc3/gigaword_eng_5_d3/data")):
        for filename in os.listdir(os.path.join(os.path.join(os.getcwd(), "disc3/gigaword_eng_5_d3/data"), dir)):
            with open(os.path.join(os.path.join(os.path.join(os.getcwd(), "disc3/gigaword_eng_5_d3/data"), dir),filename), "r", encoding="utf8") as f:
                content = f.read()
                for c in constrs:
                    k=0
                    while k != -1:
                        k = content.find(c, k+1)
                        if k != -1:
                            output.append(content[content.rfind("<P>", 0,k)+4:content.find("</P>", k)] + "\n")
                            print(output[-1])
                            counts[c] += content.count(c)
    print(counts)
    with open("results.txt", "w") as f:
        f.writelines(output)



if __name__ == '__main__':
    main()
