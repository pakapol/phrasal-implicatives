import sys, os, itertools
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #dirs = ["standard", "overlap", "disjoint"]
    #dir_data = []

    #for dname in dirs:
    cats = {}
    with open("../results/{}_conf_log".format(sys.argv[1])) as f: # log-cat-...
        for line in f:
            splitted = line[:-1].split(" ")
            cat = splitted[0]
            numbers = map(float, splitted[1:])
            cats[cat] = numbers
    l = ['entails','contradicts','permits']
    reordered_cats = ["{}=>{}".format(x,y) for x,y in itertools.product(l,l)] # insert reordering here

    reordered_array = np.array([cats[i] for i in reordered_cats])
    plt.imshow(reordered_array, cmap='RdYlBu') # RdBu, RdYlBu Spectral better?
    plt.yticks(range(len(reordered_cats)), reordered_cats)
    plt.show()
