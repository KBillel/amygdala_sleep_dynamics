from bk import io
from bk import plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

from settings import colors

def norm_baseline(activity,norm):
    baseline = np.nanmean(activity[:,norm[0]:norm[1]],1)
    activity = (activity.T / baseline).T
    activity *= 100
    return activity

def plot_activity_at_transitions(activity, metadata,stru,quintile=None,norm = None,ax = None):
    mask_stru = metadata.Region == stru
    metadata = metadata[mask_stru].copy()
    activity = activity[mask_stru].copy()
    mask_pyr = metadata.Type == 'Pyr'
    mask_int = metadata.Type == 'Int'

    quintile_labels = ['VL','L','M','H','VH']
    
    if ax is None:
        fig,ax = plt.subplots()
    
    if norm is None:
        pass
    elif isinstance(norm,list):
        activity = norm_baseline(activity,norm)
    elif norm.lower() == 'zscore':
        activity = zscore(activity,1)
    mask_finite = np.isfinite(np.nanmean(activity,1))
    activity = activity[mask_finite]
    metadata = metadata[mask_finite]
    mask_pyr = mask_pyr[mask_finite]
    mask_int = mask_int[mask_finite]
    if quintile is not None:
        metadata.loc[mask_pyr,'Quintile'] = pd.qcut(metadata[mask_pyr][quintile],q = 5,labels=quintile_labels)
        for q in np.unique(metadata.loc[mask_pyr,'Quintile']):
            mask_q = metadata.Quintile == q
            y_ = np.nanmean(activity[mask_pyr & mask_q],0)
            ax.plot(y_,c = colors[q])
    else:
        y_ = np.nanmean(activity[mask_pyr],0)
        ax.plot(y_,c = colors[stru])
        
    
    ax.plot(np.nanmean(activity[mask_int],0),'k--')


    return metadata


if __name__ == '__main__':
    plt.ion()
    transitions = io.load_shelve('processed_data/transitions')['merged_sessions']
    df_firing_rates = pd.read_csv('processed_data/states_fr.csv')

    transitions_of_interest = ['NREM-REM',
                               'REM-NREM',
                               'WAKE_HOMECAGE-NREM',
                               'REM-WAKE_HOMECAGE']

    fig,ax = plt.subplots(3,len(transitions_of_interest),figsize = (16,8))
    for i,transition_name in enumerate(transitions_of_interest):
        c_transitions = transitions[transition_name]
        c_activity = c_transitions['activity']
        c_metadata = pd.merge(c_transitions['metadata'],df_firing_rates,on = ['Rat','Day','Shank','Id','Region','Type'],how='left')

        plot_activity_at_transitions(c_activity,c_metadata,'BLA',norm = None,quintile='SLEEP',ax=ax[0,i])
        plot_activity_at_transitions(c_activity,c_metadata,'BLA',norm = 'zscore',quintile='SLEEP',ax=ax[1,i])
        plot_activity_at_transitions(c_activity,c_metadata,'BLA',norm = [20,30],quintile='SLEEP',ax=ax[2,i])

        
        ax[0,i].set_title(transition_name)
    
    for i in ax:
        for j in i: 
            plot.clean_axes(j)

    ax[0,0].set_ylabel('FR')
    ax[1,0].set_ylabel('FR(zscore)')
    ax[2,0].set_ylabel('FR %baseline')

    plt.tight_layout()
