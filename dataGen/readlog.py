import sys
import numpy as np
import matplotlib.pyplot as plt

class PITrainingLog(object):
    def __init__(self):
        self.trainloss = []
        self.trainacc = []
        self.valloss = []
        self.valacc = []
        self.lr = None
        self.init_scale = None

def readlog(filename):
    examples = []
    with open(filename, 'r') as f:
        entirelog = f.read().split('\n')
        newlog = None
        for i in xrange(len(entirelog)):
            line = entirelog[i]
            if line  == "::: Begin Log :::":
                newlog = PITrainingLog()
            elif "init_scale: " in line:
                newlog.init_scale = float(line[len("init_scale: "):])
            elif "learning_rate: " in line:
                newlog.lr = float(line[len("learning_rate: "):])
            elif "train loss:" in line:
                newlog.trainloss = [float(x) for x in entirelog[i+1].split()]
            elif "train acc:" in line:
                newlog.trainacc = [float(x) for x in entirelog[i+1].split()]
            elif "val loss:" in line:
                newlog.valloss = [float(x) for x in entirelog[i+1].split()]
            elif "val acc:" in line:
                newlog.valacc = [float(x) for x in entirelog[i+1].split()]
            elif line == "::: End of Log :::":
                examples.append(newlog)
    return examples

def scatterPlot(logs, epoch):
    init_scale_s = np.array([l.init_scale for l in logs])
    lr_s = np.array([l.lr for l in logs])
    val_loss_s = np.array([l.valloss[epoch] for l in logs])
    val_acc_s = np.array([l.valacc[epoch] for l in logs])

    ### change c=val_acc_s to c=val_loss_s to display loss function instead
    plt.scatter(init_scale_s, lr_s, c=val_acc_s, s=100, cmap='jet_r')

    ### log scale
    plt.xscale('log')
    plt.yscale('log')

    ### change the limit of the axis
    plt.xlim([10 ** -3.0, 10 ** 1.0])
    plt.ylim([10 ** -3.0, 10 ** 1.0])

    plt.show()

def getLearningCurveNear(ws, lr, logs):
    closest = sorted(logs, key=lambda x: np.log(x.init_scale/ws) ** 2 + np.log(x.lr/lr) ** 2)[0]

    ### change closest.valacc to closest.valloss to display loss function instead
    plt.plot(closest.valacc)

    plt.show()

if __name__=='__main__':
    scatterPlot(readlog(sys.argv[1]), int(sys.argv[2]))
