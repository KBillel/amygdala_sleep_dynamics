from settings import colors

from bk import io
from bk import plot

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from statannotations.Annotator import Annotator
from itertools import combinations
from scipy.stats import kruskal,zscore
from plots.plot_transitions import plot_all_network_metrics
import json
def load_data(path):
    return io.load_shelve(path)

def plot_examples(data,start,stop,ax):

    states = data['metrics']['examples']['States']
    for s in states:
        states[s] = states[s].drop('state',axis = 1)
    
    t_fr = data['metrics']['examples']['FR'][0]
    pyr_fr = zscore(data['metrics']['examples']['FR'][1])
    int_fr = zscore(data['metrics']['examples']['FR'][2])

    t_eib = data['metrics']['raw']['z_eib'][0]
    eib = data['metrics']['raw']['z_eib'][1]

    t_cv = data['metrics']['raw']['cv'][0]
    cv = zscore(data['metrics']['raw']['cv'][1])

    t_sync = data['metrics']['raw']['sync'][0]
    sync = zscore(data['metrics']['raw']['sync'][1])


    for state in ['NREM','REM','WAKE_HOMECAGE']:
        plot.intervals(states[state],col = colors[state],ax= ax)

    ax.plot(t_fr,pyr_fr,c = colors['BLA']['H'])
    ax.plot(t_fr,int_fr-5,'k--')
    ax.plot(t_eib,eib-10,colors['EIB'])
    ax.plot(t_cv,cv-15,colors['CV'])
    ax.plot(t_sync,sync-20,colors['sync'])

    ax.set_xlabel('Time (s)')

def plot_nbins(data,ax = None):
    ylim = {'z_eib':(-1,1),
            'cv':(1,2.5),
            'sync':(0,0.015)}

    if ax is None:
        fig,ax = plt.subplots(len(data),3,figsize = (4,8),sharey='row')
    for i,(metric_name,all_states) in enumerate(data.items()):
        for j,(state,values) in enumerate(all_states.items()):
            x = range(len(values.T))
            plot.confidence_intervals(x,values,ax = ax[i,j],style=colors[state],alpha=0.2)
            ax[i,j].set_xlabel('Bins')
            ax[i,j].set_ylim(ylim[metric_name])
            ax[i,j].set_title(state)


def plot_epochs(data,ax = None):
    if ax is None:
        fig,ax = plt.subplots(1,len(data),figsize = (16,8))

    for i,(metric_name,all_states) in enumerate(data.items()):
        df = metric_epoch_to_df(metric_name, all_states)
        if len(np.unique(df.times)) == 1:
            plotting_params = {'data':df,
                               'x':'state',
                               'y':metric_name,
                               'ax':ax[i],
                               'palette':colors}
            
            
            states = ['REM','NREM','WAKE_HOMECAGE']
            pairs = list(combinations(states,2))
            sns.boxenplot(**plotting_params)
            annotator = Annotator(pairs = pairs,**plotting_params)
            _,stats_data = annotator.configure(test="Mann-Whitney",comparisons_correction = 'Bonferroni').apply_and_annotate()
            kruskal_stats = kruskal(*[df[df['state'] == s][metric_name] for s in np.unique(df.state)])
            print(metric_name,kruskal_stats)
        else:
            df.times[df.times==0] = 'First'
            df.times[df.times==1] = 'Middle'
            df.times[df.times==2] = 'Last'
            pairs = [
                    [('NREM', 'First'), ('NREM', 'Middle')],
                    [('NREM', 'First'), ('NREM', 'Last')],
                    [('NREM', 'Middle'), ('NREM', 'Last')],

                    [('REM', 'First'), ('REM', 'Middle')],
                    [('REM', 'First'), ('REM', 'Last')],
                    [('REM', 'Middle'), ('REM', 'Last')],

                    [('WAKE_HOMECAGE', 'First'), ('WAKE_HOMECAGE', 'Middle')],
                    [('WAKE_HOMECAGE', 'First'), ('WAKE_HOMECAGE', 'Last')],
                    [('WAKE_HOMECAGE', 'Middle'), ('WAKE_HOMECAGE', 'Last')]
                    ]
            print(i)
            plotting_params = {'data':df,
                                'x':'state',
                                'y':metric_name,
                                'ax':ax[i],
                                'hue':'times',
                                'color':'g'}
            print(f'Metric Name:{metric_name}')
            sns.boxenplot(**plotting_params)
            annotator = Annotator(pairs = pairs,**plotting_params)
            _,stats_data = annotator.configure(test="Wilcoxon",comparisons_correction = 'Bonferroni').apply_and_annotate()


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

def fig5_network_metrics(data):
    fig,ax = plt.subplot_mosaic('''AAA
                                   BCD''',figsize = (12,8))
    plot_examples(data['unique_sessions'][example_session],0,20000,ax['A'])
    plot_epochs(data['merged_sessions']['epochs'],(ax['B'],ax['C'],ax['D']))

    # plot_epochs(data['merged_sessions']['thirds'],(ax['B'],ax['C'],ax['D']))
    # plot_nbins(data['merged_sessions']['nbins'],ax[:,2:])
    for a in ax.values(): plot.clean_axes(a)
    ax['A'].set_xlim(12500,12500+3600*2)
    ax['A'].set_xticks(np.arange(12500,12500+3600*2,1000),np.arange(0,7200,1000))
    # ax['B'].set_ylim(-3,5)
    # ax['C'].set_ylim(0,6)
    # ax['D'].set_ylim(-0.02,0.08)
    plot.labels_in_grid(list(ax.values()),numbers={
                                                   (0,0):0,
                                                   (1,0):1,
                                                   (1,1):2,
                                                   (1,2):3},all_ylabels=True)
    fig.tight_layout()
    fig.savefig('plots/figures/main-network_metrics_states.png')
    fig.savefig('plots/figures/main-network_metrics_states.svg')
    return fig,ax


def fig6_network_metrics(data,params):
    transitions_of_interest = {'NREM':[0,30],
                           'REM':[0,12],
                           'WAKE_HOMECAGE':[0,30]}
    
    # fig,ax = plt.subplot_mosaic('''AAA
    
    #                                BCD''',figsize = (12,8))
    fig,ax = plt.subplots(2,3,figsize = (12,8))
    # plot_nbins(data['merged_sessions']['nbins'])
    
    
    plot_all_network_metrics(data['merged_sessions']['at_transitions'],transitions_of_interest,params,ax[0,:])
    for _,d in data['merged_sessions'].items():
        for m in metrics_to_pop: d.pop(m)
    
    plot_epochs(data['merged_sessions']['thirds'],(ax[1,0],ax[1,1],ax[1,2]))
    for a in ax.flatten(): plot.clean_axes(a)
    # ax['B'].set_ylim(-3,5)
    # ax['C'].set_ylim(0,6)
    # ax['D'].set_ylim(-0.02,0.08)
    plot.labels_in_grid(ax.flatten(),numbers={
                                                   (0,0):0,
                                                   (1,0):1,
                                                   (1,1):2,
                                                   (1,2):3},all_ylabels=True)
    
    for i,title in enumerate(transitions_of_interest.keys()):
        ax[0,i].set_title(title)
    for a in ax[0,:]: a.set_ylim(-1,1)
    
    fig.tight_layout()
    fig.savefig('plots/figures/main-network_metrics_bins.png')
    fig.savefig('plots/figures/main-network_metrics_bins.svg')
    return fig,ax
if __name__ == '__main__':

    example_session = 'Rat09-20140401'
    plt.ion()
    data = load_data('processed_data/network_metrics')
    with open('processed_data/transitions.json','r') as jf:
        params = json.load(jf)
    metrics_to_pop = ['eib']
    for _,d in data['merged_sessions'].items():
        for m in metrics_to_pop: d.pop(m)

    fig,ax = fig5_network_metrics(data)
    # 
    data = load_data('processed_data/network_metrics')
    fig6_network_metrics(data,params)
    # print(ax.values())


