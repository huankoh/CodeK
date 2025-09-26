import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import os

def plot_roc_curve(fpr, tpr, auc, title='Model', savefig=None, show=False):
    plt.plot(fpr, tpr,
             label='{} ROC Curve (area ={:.3f})'.format(title, auc))
    plt.legend(loc='best')
    if savefig is not None:
        plt.savefig(f'{savefig}_{title}_roc.svg')
    if show:
        plt.show()
    plt.clf()
    plt.close()

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)