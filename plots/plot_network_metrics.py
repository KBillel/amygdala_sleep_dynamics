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

    ax.plot(t_fr,pyr_fr,colors['BLA']['BLA'])
    ax.plot(t_fr,int_fr-5,'k--')
    ax.plot(t_eib,eib-10,colors['EIB'])
    ax.plot(t_cv,cv-15,colors['CV'])
    ax.plot(t_sync,sync-20,colors['sync'])

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
            
            plotting_params = {'data':df,
                                'x':'state',
                                'y':metric_name,
                                'ax':ax[i],
                                'hue':'times'}
            
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

if __name__ == '__main__':

    example_session = 'Rat08-20130717'

    plt.ion()
    data = load_data('processed_data/network_metrics')
    metrics_to_pop = ['eib']

    for _,d in data['merged_sessions'].items():
        for m in metrics_to_pop: d.pop(m)

    # gridspec_kw={'width_ratios':[3,6,2,2,2]}
    # fig,ax = plt.subplots(2,3,sharex='col',figsize = (12,8),squeeze=False)
    fig,ax = plt.subplot_mosaic('''
                                AAA
                                BCD''',figsize = (12,8))

    # for i in range(len(ax)):
    #     ax[i,2].sharey(ax[i,3])
    #     ax[i,3].sharey(ax[i,4])
    plot_examples(data['unique_sessions'][example_session],0,20000,ax['A'])
    plot_epochs(data['merged_sessions']['epochs'],(ax['B'],ax['C'],ax['D']))
    # plot_epochs(data['merged_sessions']['thirds'],ax = ax[1,:])
    # plot_nbins(data['merged_sessions']['nbins'],ax[:,2:])
    for a in ax.values(): plot.clean_axes(a)
    ax['A'].set_xlim(2000,2000+3600)
    ax['B'].set_ylim(-3,5)
    ax['C'].set_ylim(0,6)
    ax['D'].set_ylim(-0.02,0.08)
    fig.tight_layout()
    plt.show()
    fig.savefig('output.png')
    fig.savefig('plots/figures/network_metrics2.svg')

    # plt.close(fig)