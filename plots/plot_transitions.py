from bk import io
from bk import plot
from bk.misc import filter_neurons
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import json

from settings import colors

def norm_baseline(activity,norm):
    baseline = np.nanmean(activity[:,norm[0]:norm[1]],1)
    activity = (activity.T / baseline).T
    activity *= 100
    return activity

def quantile_cut(df,on,quantile_labels = ('VL','L','M','H','VH')):
    df = df.copy()
    q = len(quantile_labels)
    df.loc[:,'Quantile'] = pd.qcut(df[on],q = q,labels=quantile_labels)
    return df


def plot_activity_at_transitions(activity, metadata,stru,bin_state,quantile=None,norm = None,ax = None):
    quantile_labels = ['VL','L','M','H','VH']
    if ax is None:
        fig,ax = plt.subplots()
    if norm is None:
        pass
    elif isinstance(norm,list):
        activity = norm_baseline(activity,norm)
    elif norm.lower() == 'zscore':
        activity = zscore(activity,1)
    
    activity_pyr,metadata_pyr = filter_neurons(activity,metadata,stru,'Pyr',finite = True)
    activity_int,metadata_int = filter_neurons(activity,metadata,stru,'Int',finite = True)

    if quantile is not None:
        metadata_pyr = quantile_cut(metadata_pyr,quantile,quantile_labels)
        for q in quantile_labels:
            mask_q = metadata_pyr['Quantile'] == q
            y_ = np.nanmean(activity_pyr[mask_q],0)
            ax.plot(y_,c = colors[q])
    else:
        y_ = np.nanmean(activity_pyr,0)
        ax.plot(y_,c = colors[stru])
        
    ax.plot(np.nanmean(activity_int,0),'k--')

    c = 0
    for state,nbins in bin_state.items():
        ax.axvspan(c,c+nbins, facecolor=colors[state], alpha=0.5)
        c+=nbins
    return metadata


if __name__ == '__main__':
    plt.ion()
    transitions = io.load_shelve('processed_data/transitions')['merged_sessions']
    df_firing_rates = pd.read_csv('processed_data/states_fr.csv')
    
    with open('processed_data/transitions.json','r') as jf:
        params = json.load(jf)

    transitions_of_interest = ['NREM',
                               'REM',
                               'WAKE_HOMECAGE',
                               'NREM-REM',
                               'REM-NREM',
                               'WAKE_HOMECAGE-NREM',
                               'REM-WAKE_HOMECAGE']

    fig,ax = plt.subplots(3,len(transitions_of_interest),figsize = (16,8),sharey='row',sharex='col')
    for i,transition_name in enumerate(transitions_of_interest):

        states = transition_name.split('-')
        bin_state = {s:params['nbins'][s] for s in states}
        
        c_transitions = transitions[transition_name]
        c_activity = c_transitions['activity']
        c_metadata = pd.merge(c_transitions['metadata'],df_firing_rates,on = ['Rat','Day','Shank','Id','Region','Type'],how='left')

        plot_activity_at_transitions(c_activity,c_metadata,'Hpc',norm = None,quantile='WAKE_HOMECAGE',ax=ax[0,i],bin_state = bin_state)
        ax[0,i].semilogy()
        plot_activity_at_transitions(c_activity,c_metadata,'Hpc',norm = 'zscore',quantile='WAKE_HOMECAGE',ax=ax[1,i],bin_state = bin_state)
        plot_activity_at_transitions(c_activity,c_metadata,'Hpc',norm = None,quantile= None,ax=ax[2,i],bin_state = bin_state)

        ax[0,i].set_title(transition_name)
    
    for i in ax:
        for j in i: 
            plot.clean_axes(j)

    ax[0,0].set_ylabel('FR')
    ax[1,0].set_ylabel('FR(zscore)')
    ax[2,0].set_ylabel('FR %baseline')

    plt.tight_layout()
