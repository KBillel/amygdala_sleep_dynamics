import json
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import zscore, linregress

from bk import io
from bk import plot
from bk.misc import filter_neurons
from settings import colors


def set_rem_labels(df):
    rem_on = df.pInc <= 0.001
    rem_off = df.pDec <= 0.001
    rem_unknown = ((df.pInc > 0.001) & (df.pDec > 0.001))

    df['REM_label'] = np.nan
    df['REM_label'][rem_on] = 'REM_ON'
    df['REM_label'][rem_off] = 'REM_OFF'
    df['REM_label'][rem_unknown] = 'Unknown'

    return df

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


def compute_corr(activity):
    # x = [range(activity.shape[1])]*len(activity)
    # reg = linregress(np.hstack(x),activity.flatten())

    x = range(activity.shape[1])
    y = np.nanmean(activity,0)
    reg = linregress(x,y)
    p = reg.pvalue*8
    if p >= 1: p = 1
    return reg.rvalue, p


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
    print(activity_pyr.shape)
    activity_int,metadata_int = filter_neurons(activity,metadata,stru,'Int',finite = True)
    max_average,min_average = [],[]
    if quantile is not None:
        colors_plt = plt.cm.Greens(np.linspace(0,1,len(quantile_labels)+1))
        metadata_pyr = quantile_cut(metadata_pyr,quantile,quantile_labels)
        for i,q in enumerate(quantile_labels):
            mask_q = metadata_pyr['Quantile'] == q
            # y_ = np.nanmean(activity_pyr[mask_q],0)
            # ax.plot(y_,c = colors_stru[q])
            plot.confidence_intervals(range(activity_pyr[mask_q].shape[1]),activity_pyr[mask_q],style = colors_plt[i],ax = ax,alpha = 0.2)
            r,p = compute_corr(activity_pyr[mask_q])
            ax.text(0.25,0.5+0.1*i,f'r = {r:.2g} p = {p:.2g}',transform = ax.transAxes, color = colors_plt[i])
            print(r,p)
            max_average.append(np.max(np.nanmean(activity_pyr[mask_q],0)))
            min_average.append(np.min(np.nanmean(activity_pyr[mask_q],0)))
    else:
        # y_ = np.nanmean(activity_pyr,0)
        # ax.plot(y_,c = colors_stru[stru])
        plot.confidence_intervals(range(activity_pyr.shape[1]),activity_pyr[metadata_pyr['REM_label'] == 'REM_ON'],style = colors_stru[stru],ax = ax,alpha = 0.2)
        plot.confidence_intervals(range(activity_pyr.shape[1]),activity_pyr[metadata_pyr['REM_label'] == 'REM_OFF'],style = 'r',ax = ax,alpha = 0.2)
        r,p = compute_corr(activity_pyr[metadata_pyr['REM_label'] == 'REM_ON'])
        ax.text(0.25,0.9,f'r = {r:.2g} p = {p:.2g}',transform = ax.transAxes, color = colors_stru[stru])
        r,p = compute_corr(activity_pyr[metadata_pyr['REM_label'] == 'REM_OFF'])
        ax.text(0.25,0.8,f'r = {r:.2g} p = {p:.2g}',transform = ax.transAxes, color = 'r')

        max_average.append(np.max(np.nanmean(activity_pyr,0)))
        min_average.append(np.min(np.nanmean(activity_pyr,0)))
        
    # ax.plot(np.nanmean(activity_int,0),'k--')
    # plot.confidence_intervals(range(activity_int.shape[1]),activity_int,style = 'k',ax = ax,alpha = 0.2)
    # r,p = compute_corr(activity_int)
    # ax.text(0.25,1,f'r = {r:.2g} p = {p:.2g}',transform = ax.transAxes, color = 'k')
    max_average.append(np.max(np.nanmean(activity_int,0)))
    min_average.append(np.min(np.nanmean(activity_int,0)))

    c = 0
    for state,nbins in bin_state:
        ax.axvspan(c,c+(nbins-1), facecolor=colors[state], alpha=0.5,zorder = -10)
        c+=nbins
    return ax,min_average,max_average


def plot_all_transitions(transitions, df_firing_rates, stru,state, params, transitions_of_interest, ax):
    
    for i,(transition_name,baseline )in enumerate(transitions_of_interest.items()):
        states = transition_name.split('-')
        bin_state = [(s,params['nbins'][s]) for s in states]
        
        c_transitions = transitions[transition_name]
        c_activity = c_transitions['activity']
        c_metadata = pd.merge(c_transitions['metadata'],df_firing_rates,on = ['Rat','Day','Shank','Id','Region','Type'],how='left')
        print(transition_name, c_transitions['n_transitions'])
        plot_activity_at_transitions(c_activity,c_metadata,stru,norm = 'zscore',quantile=None,ax=ax[0,i],bin_state = bin_state)
        plot_activity_at_transitions(c_activity,c_metadata,stru,norm = 'None',quantile=state,ax=ax[1,i],bin_state = bin_state)
        ax[1,i].semilogy()
        plot_activity_at_transitions(c_activity,c_metadata,stru,norm = 'zscore',quantile=state,ax=ax[2,i],bin_state = bin_state)
        ax[2,i].set_xlabel('Times (bins)')

        # ax_baseline,ymin,ymax = plot_activity_at_transitions(c_activity,c_metadata,stru,norm = baseline,quantile= state,ax=ax[2,i],bin_state = bin_state)
        # ymin = min(ymin)*0.8
        # ymax = max(ymax)*1.2
        # old_ylim = ax[2,0].get_ylim()
        # print(old_ylim)
        # new_ymin = min(ymin,old_ylim[0])
        # new_ymax = max(ymax,old_ylim[1])

        # ax[2,0].set_ylim(new_ymin,new_ymax)
        ax[0,i].set_title(transition_name)
    
    for i in ax.flatten():
        plot.clean_axes(i)
    
    
    ax[0,0].set_ylabel('FR (zscore)')
    ax[1,0].set_ylabel('FR (Hz)')
    ax[2,0].set_ylabel('FR (zscore)')
    # ax[1,0].set_ylim(-1.5,1.5)
    # ax[2,0].set_ylim(75,250)
    plt.tight_layout()

def plot_network_metrics(values,norm = True,col = 'b',ax = None):
    if ax is None:
        fig,ax = plt.subplots()
    if norm:
        values = zscore(values,1)
    x = range(values.shape[1])
    plot.confidence_intervals(x,values,style = col,ax = ax)
    r,p = compute_corr(values)
    return r,p*9
    



def plot_all_network_metrics(network_metrics_transitions,transitions_of_interest,params,ax = None):
    

    for i,(transition_name,baseline )in enumerate(transitions_of_interest.items()):
        states = transition_name.split('-')
        bin_state = [(s,params['nbins'][s]) for s in states]
        r,p = plot_network_metrics(network_metrics_transitions['sync'][transition_name],True,col = colors['sync'],ax = ax[i])
        ax[i].text(0.25,0.8,f'r = {r:.2g} p = {p:.2g}',transform = ax[i].transAxes, color = colors['sync'])
        r,p = plot_network_metrics(network_metrics_transitions['eib'][transition_name],True,col = colors['EIB'],ax = ax[i])
        ax[i].text(0.25,0.9,f'r = {r:.2g} p = {p:.2g}',transform = ax[i].transAxes, color = colors['EIB'])
        r,p = plot_network_metrics(network_metrics_transitions['cv'][transition_name],True,col = colors['CV'],ax = ax[i])
        ax[i].text(0.25,1,f'r = {r:.2g} p = {p:.2g}',transform = ax[i].transAxes, color = colors['CV'])

        ax[i].set_xlabel('Times (bins)')
        c = 0
        for state,nbins in bin_state:
            ax[i].axvspan(c,c+(nbins-1), facecolor=colors[state], alpha=0.5,zorder = -10)
            c+=nbins
        for i in ax.flatten():
            plot.clean_axes(i)


def plot_histogram_fr(df,stru,params,ax):
    df_stru_pyr = df[(df.Region == stru) & (df.Type == 'Pyr')]
    logbins = np.logspace(-3,3,100)
    for i,state in enumerate(['NREM','REM','WAKE_HOMECAGE']):
        q,bins = pd.qcut(df_stru_pyr[state],5,retbins=True)

        for b in bins[1:-1]:
            ax[i].axvline(b,c = 'r')
        ax[i].hist(df_stru_pyr[state],logbins,facecolor = colors[state])
        ax[i].semilogx()
        ax[i].set_title(state)
        ax[i].set_xlabel('Firing Rates (Hz)')
        ax[i].set_ylabel('Counts')



if __name__ == '__main__':

    stru =  'BLA'
    plt.ion()
    transitions = io.load_shelve('processed_data/transitions')['merged_sessions']
    network_metrics_transitions = io.load_shelve('processed_data/network_metrics')['merged_sessions']['at_transitions']
    df_firing_rates = pd.read_csv('processed_data/states_fr.csv')
    df_rem_on = pd.read_csv('processed_data/rem_on.csv')
    df_rem_on = set_rem_labels(df_rem_on)
    df_firing_rates = pd.merge(df_firing_rates,df_rem_on,on=['Rat','Day','Shank','Id','Region','Type','SessID'])

    with open('processed_data/transitions.json','r') as jf:
        params = json.load(jf)

    transitions_of_interest = {
                               'WAKE_HOMECAGE':[0,30],
                               'NREM':[0,30],
                               'REM':[0,12],
                            'NREM-REM':[20,30]
                            #    'WAKE_HOMECAGE-NREM':[20,30],
                            #    'NREM-REM':[20,30],
                            #    'REM-NREM':[8,12],
                            # 'NREM-REM-NREM':[1,2],
                            # 'REM-NREM-REM':[1,2],
                            # 'NREM-REM-WAKE_HOMECAGE':[1,2]
                            #    'REM-WAKE_HOMECAGE':[8,12],
                            #    'WAKE_HOMECAGE-extended_sleep-WAKE_HOMECAGE':[50,60],
                            # 'REM-NREM-REM':[22,32]
                            }

    
    quantiles_states = ['WAKE_HOMECAGE']
    for state in quantiles_states:
        fig,ax = plt.subplots(3,len(transitions_of_interest),figsize = (12,8),sharey='row',sharex='col',squeeze=False)
        
        plot_all_transitions(transitions, df_firing_rates,stru,state, params, transitions_of_interest, ax)
        # plot_all_network_metrics(network_metrics_transitions,transitions_of_interest,params,ax = ax[2,:])
        ax[0,0].set_ylim(-1,1)

        ax[2,0].set_ylim(-1,1)
        fig.tight_layout()
        fig.savefig(f'plots/figures/transition_epochs-sleep-z.svg')
        fig.savefig('output.png')
        

    # fig,ax = plt.subplots(1,3,figsize = (12,4),sharey = True)
    # plot_histogram_fr(df_firing_rates,stru,params,ax)
    # for axe in ax.flatten():
    #     plot.forceAspect(axe)
    #     plot.clean_axes(axe)
    # fig.tight_layout()
    # fig.savefig('plots/figures/histograms.svg')
    # plt.show()
