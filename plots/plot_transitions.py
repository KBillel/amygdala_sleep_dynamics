from bk import io
from bk import plot
from bk.misc import filter_neurons
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import json

from settings import colors

from typing import Union, Optional, Tuple, Dict, Sequence
from numpy.typing import ArrayLike

def norm_baseline(activity:ArrayLike,norm:list[int])->ArrayLike:
    """
    Normalize each neuron in activity on the period given by norm

    Parameters
    ----------
    activity : ArrayLike
        [neurons,times] firing rates of the neurons see :py:func:`compute_transitions_activity`
    norm : list[int]
        baseline interval to compute the normalizatin

    Returns
    -------
    ArrayLike
        Normalized Activity
    """
    baseline = np.nanmean(activity[:,norm[0]:norm[1]],1)
    activity = (activity.T / baseline).T
    activity *= 100
    return activity

def quantile_cut(df:pd.DataFrame,on:str,quantile_labels:Sequence = ('VL','L','M','H','VH'))->pd.DataFrame:
    """
    Cut the population in quantiles based on a single variable.

    Parameters
    ----------
    df : pd.DataFrame
        metadata + data dataframe
    on : str
        variable to cut the population on
    quantile_labels : Sequence, optional
        labels of each quantile (here VeryLow,Low,Medium,High and VeryHigh), by default ('VL','L','M','H','VH')

    Returns
    -------
    pd.DataFrame
        df with added Quantile column
    """
    df = df.copy()
    q = len(quantile_labels)
    df.loc[:,'Quantile'] = pd.qcut(df[on],q = q,labels=quantile_labels)
    return df


def plot_activity_at_transitions(activity, metadata,stru,bin_state,quantile=None,norm = None,ax = None):

    colors_stru = colors[stru]

    quantile_labels = ['VL','L','M','H','VH']
    # quantile_labels = [i for i in range(10)]
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
    print(activity_int.shape)
    max_average,min_average = [],[]
    if quantile is not None:
        colors_plt = plt.cm.Greens(np.linspace(0,1,len(quantile_labels)+1))
        metadata_pyr = quantile_cut(metadata_pyr,quantile,quantile_labels)
        for i,q in enumerate(quantile_labels):
            mask_q = metadata_pyr['Quantile'] == q
            # y_ = np.nanmean(activity_pyr[mask_q],0)
            # ax.plot(y_,c = colors_stru[q])
            plot.confidence_intervals(range(activity_pyr[mask_q].shape[1]),activity_pyr[mask_q],style = colors_plt[i],ax = ax,alpha = 0.2)
            max_average.append(np.max(np.nanmean(activity_pyr[mask_q],0)))
            min_average.append(np.min(np.nanmean(activity_pyr[mask_q],0)))
    else:
        # y_ = np.nanmean(activity_pyr,0)
        # ax.plot(y_,c = colors_stru[stru])
        plot.confidence_intervals(range(activity_pyr.shape[1]),activity_pyr,style = colors_stru[stru],ax = ax,alpha = 0.2)
        max_average.append(np.max(np.nanmean(activity_pyr,0)))
        min_average.append(np.min(np.nanmean(activity_pyr,0)))
        
    # ax.plot(np.nanmean(activity_int,0),'k--')
    plot.confidence_intervals(range(activity_int.shape[1]),activity_int,style = 'k',ax = ax,alpha = 0.2)
    max_average.append(np.max(np.nanmean(activity_int,0)))
    min_average.append(np.min(np.nanmean(activity_int,0)))

    c = 0
    for state,nbins in bin_state:
        ax.axvspan(c,c+nbins, facecolor=colors[state], alpha=0.5,zorder = -10)
        c+=nbins
    return ax,min_average,max_average


def plot_all_transitions(transitions, df_firing_rates, stru,state, params, transitions_of_interest, ax):
    
    for i,(transition_name,baseline )in enumerate(transitions_of_interest.items()):
        states = transition_name.split('-')
        bin_state = [(s,params['nbins'][s]) for s in states]
        
        c_transitions = transitions[transition_name]
        c_activity = c_transitions['activity']
        c_metadata = pd.merge(c_transitions['metadata'],df_firing_rates,on = ['Rat','Day','Shank','Id','Region','Type'],how='left')

        plot_activity_at_transitions(c_activity,c_metadata,stru,norm = 'None',quantile=state,ax=ax[0,i],bin_state = bin_state)
        ax[0,i].semilogy()
        plot_activity_at_transitions(c_activity,c_metadata,stru,norm = 'zscore',quantile=state,ax=ax[1,i],bin_state = bin_state)
        ax_baseline,ymin,ymax = plot_activity_at_transitions(c_activity,c_metadata,stru,norm = baseline,quantile= state,ax=ax[2,i],bin_state = bin_state)
        ymin = min(ymin)*0.8
        ymax = max(ymax)*1.2
        old_ylim = ax[2,0].get_ylim()
        print(old_ylim)
        new_ymin = min(ymin,old_ylim[0])
        new_ymax = max(ymax,old_ylim[1])

        ax[2,0].set_ylim(new_ymin,new_ymax)
        ax[0,i].set_title(transition_name)
    
    for i in ax.flatten():
        plot.clean_axes(i)
    
    
    
    ax[0,0].set_ylabel('FR')
    ax[1,0].set_ylabel('FR(zscore)')
    ax[2,0].set_ylabel('FR % Baseline')
    ax[1,0].set_ylim(-1.5,1.5)
    # ax[2,0].set_ylim(75,250)
    plt.tight_layout()

if __name__ == '__main__':

    stru =  'BLA'
    plt.ion()
    transitions = io.load_shelve('processed_data/transitions')['merged_sessions']
    df_firing_rates = pd.read_csv('processed_data/states_fr.csv')
    with open('processed_data/transitions.json','r') as jf:
        params = json.load(jf)

    transitions_of_interest = {
                            #    'NREM':[0,30],
                            #    'REM':[0,12],
                            #    'WAKE_HOMECAGE':[0,30],
                            #    'NREM-REM':[20,30],
                            #    'REM-NREM':[8,12],
                            #    'WAKE_HOMECAGE-NREM-REM':[20,30],
                            #    'REM-WAKE_HOMECAGE':[8,12],
                               'WAKE_HOMECAGE-extended_sleep-WAKE_HOMECAGE':[50,60],
                            # 'REM-NREM-REM':[22,32]
                            }

    
    quantiles_states = ['WAKE_HOMECAGE','SLEEP']
    for state in quantiles_states:
        fig,ax = plt.subplots(3,len(transitions_of_interest),figsize = (16,8),sharey='row',sharex='col',squeeze=False)
        
        plot_all_transitions(transitions, df_firing_rates,stru,state, params, transitions_of_interest, ax)
        fig.savefig(f'plots/figures/transitions_ES-{state}.svg')
        # plt.close(fig)