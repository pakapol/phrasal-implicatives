import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dirs = ["standard", "overlap", "disjoint"]
    dir_data = []

    for dname in dirs:
        metrics = {}
        with open("../results/{}_conf_log".format(dname)) as f:
            for line in f:
                splitted = line[:-1].split(" ")
                metric = splitted[0]
                numbers = map(float, splitted[1:])
                metrics[metric] = numbers

        dir_data.append(metrics)
   
    f, ax = plt.subplots(3, sharex=True, figsize=(8,9))

    for i in range(3):
        p = []
        for m in metrics:
            p.append(ax[i].plot(dir_data[i][m], label=m.replace("_"," ")))
        ax[i].legend(loc=4)
        ax[i].set_title("F1 scores by epoch for {} dataset, retrained embeddings".format(dirs[i]))
    plt.show() 
