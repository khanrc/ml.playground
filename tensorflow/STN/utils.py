import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import collections


def one_hot(dense, ndim=10):
    N = dense.shape[0]
    ret = np.zeros([N, ndim])
    ret[np.arange(N), dense.reshape([-1])] = 1
    return ret


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(40, 40), cmap='Greys_r')

    return fig


def count_params():
    # total_parameters = 0
    counter = collections.defaultdict(lambda: 0)
    
    #iterating over all variables
    for variable in tf.trainable_variables():
        scope = variable.name.split('/')[0]
        local_params = np.prod(variable.shape).value
        counter[scope] += local_params
        
    # total = sum(counter.values())
    return counter