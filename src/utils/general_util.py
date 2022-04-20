import pickle
import random
import torch
import numpy as np
import matplotlib.pyplot as plt


def save_pickle(data, fname):
    '''Save data to pickle file'''
    with open(f'{fname}.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=4)


def load_pickle(fname):
    '''Load data from pickle file'''
    with open(f'{fname}.pkl', 'rb') as f:
        return pickle.load(f)


def set_seed(seed=None, seed_torch=True):
    # Adapated from NMA-DL tutorials (https://deeplearning.neuromatch.io/)

    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    print(f'Random seed {seed} has been set.')


def seed_worker(worker_id):
    # In case that `DataLoader` is used
    # Adapated from NMA-DL tutorials (https://deeplearning.neuromatch.io/)

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def set_device():
    # Adapated from NMA-DL tutorials (https://deeplearning.neuromatch.io/)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("GPU is not enabled in this notebook.")
    else:
        print("GPU is enabled in this notebook.")
    return device


def map_list_with_dict(ls, d):
    '''Map list as key to dictionery values'''
    return list(map(d.get, ls))


def make_color_map(cmap_name, labels, plot_cmap=True):
    '''
        Make custom color map 
        Returns a dictionary of label name with color value 
    '''
    cmap = plt.get_cmap(cmap_name)
    color_list = cmap(np.linspace(0, 1.0, len(labels)+1))
    colormap = dict(zip(labels, color_list[:-1]))

    if plot_cmap:
        for i, (name, c) in enumerate(colormap.items()):
            plt.axhline(-i, linewidth=10, c=c, label=name)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)

    return colormap