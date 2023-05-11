from settings import colors

from bk import io
from bk import plot

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def load_data(path):
    return io.load_shelve(path)


def plot_nbins(data,ax = None):
    if ax is None:
        fig,ax = plt.subplots(len(data),3,figsize = (4,8),sharey='row')
    for i,(metric_name,all_states) in enumerate(data.items()):
        for j,(state,values) in enumerate(all_states.items()):
            x = range(len(values.T))
            plot.confidence_intervals(x,values,ax = ax[i,j],style=colors[state])



def plot_epochs(data,ax = None):
    if ax is None:
        fig,ax = plt.subplots(1,len(data),figsize = (16,8))

    for i,(metric_name,all_states) in enumerate(data.items()):
        df = metric_epoch_to_df(metric_name, all_states)
        if len(np.unique(df.times)) == 1:
            sns.boxenplot(data = df,x = 'state',y = metric_name,ax = ax[i],palette=colors)
        else:
            sns.boxenplot(data = df,x = 'state',y = metric_name,ax = ax[i],hue = 'times')
        ax[i].legend()


def metric_epoch_to_df(metric_name, all_states):
    values = []
    states = []
    for state,value in all_states.items():
        values.extend(value)
        states.extend([state]*len(value))
    values = np.array(values)
    states = np.array(states)
    df = pd.DataFrame(values)
    df['state'] = states
    df = df.melt(id_vars = 'state',var_name='times',value_name=metric_name)
    
    return df

if __name__ == '__main__':
    plt.ion()
    data = load_data('processed_data/network_metrics')
    metrics_to_pop = ['eib']

    for _,d in data['merged_sessions'].items():
        for m in metrics_to_pop: d.pop(m)


    fig,ax = plt.subplots(3,5,sharex='col',figsize = (16,8))

    for i in range(len(ax)):
        print('y')
        ax[i,2].sharey(ax[i,3])
        ax[i,3].sharey(ax[i,4])

    plot_epochs(data['merged_sessions']['epochs'],ax[:,0])
    plot_epochs(data['merged_sessions']['thirds'],ax = ax[:,1])
    plot_nbins(data['merged_sessions']['nbins'],ax[:,2:])
    for a in ax.flatten(): plot.clean_axes(a)
    plt.tight_layout()
    plt.show()
    plt.savefig('plots/figures/network_metrics.svg')