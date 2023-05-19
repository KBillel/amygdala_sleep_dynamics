import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import linregress
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from bk import plot
from bk import io

import neuroseries as nts

from typing import Union, Optional,Tuple, Dict, Sequence
from settings import colors
def average_by_equal_sized_group(df:pd.DataFrame,nbins:int)->Tuple[pd.DataFrame]:
    """
    Average time multi dimentional time series by equal sized groups

    Parameters
    ----------
    df : pd.DataFrame
        Multi dimential time series
    nbins : int
        number of equal size group to be made

    Returns
    -------
    Tuple[pd.DataFrame]
        Times,Average
    """
    df = df.sort_values('Times')
    df.reset_index(inplace=True)
    df['bin'] = pd.qcut(df.index.values,nbins,range(nbins))
    return df.groupby('bin').mean(numeric_only = True).Times,df.groupby('bin').mean(numeric_only = True).FR,df.groupby('bin').sem(numeric_only = True).FR


def plot_scatter_extended(extended_state:list,color:str,ax:Tuple)-> bool:
    """
    Scatter plot an extended_state

    Parameters
    ----------
    extended_state : list
        list of extended given by :py:func:`fr_across_extended`
    color : str
        Code for color
    ax : _type_
        matplotlib axis

    Returns
    -------
    
    """

    df = pd.concat(extended_state,1)
    df['Times'] = df.index.values
    df_melt = df.melt(id_vars='Times',value_name='FR')
    df_melt.dropna(inplace=True)
    df = df.drop('Times',axis = 1)
    reg = scipy.stats.linregress(df_melt.Times/1_000_000,df_melt.FR)
    x = range(6000)
    y = x*reg.slope + reg.intercept 

    for i in df:
        # ax[0].plot(df.index/1_000_000,df[i],c = color,alpha = 0.3,lw = 0.5)
        ax[0].scatter(df.index/1_000_000,df[i],c = color,alpha = 0.3,s = 0.5)
    
    ax[0].plot(x,y,'k--')
    
    ax[0].text(1,1,f'r = {reg.rvalue:.4f} \np = {reg.pvalue:.4f}',transform = ax[0].transAxes)
    ax[0].set_ylim(-0.75,0.75)
    ax[0].set_xlim(-500,6000)
    plot.forceAspect(ax[0])
    ax[0].set_xlabel('Time from start of ext. sleep (s)')
    ax[0].set_ylabel('Firing rates (Z)')

    t,average,sem = average_by_equal_sized_group(df_melt,10)


    ax[1].errorbar(t/1_000_000,average,sem,linestyle='None',marker = '.',color = color,barsabove=True)
    ax[1].plot(x,y,'k--')
    ax[1].set_ylim(-0.055,0.055)
    ax[1].sharex(ax[0])
    plot.forceAspect(ax[1])
    return ax


def plot_delta(ax=None,colors = None, fr_csv='processed_data/delta_extended.csv'):
    """
    Plot barplot of delta FR beg vs end of extended sleep

    Parameters
    ----------
    ax : axis
        matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots()
    df = pd.read_csv(fr_csv)
    sns.barplot(df[(df.Region == 'BLA') & (df.Type == 'Pyr')],order=['NREM','REM','WAKE_HOMECAGE'],ax = ax,palette = colors)
    ax.set_ylim(-0.2,0.2)
    # plot.forceAspect(ax)

def average_extended(extended, stru, types):
    averaged_extended = {}
    for substate,extended_substate in extended.items():
        averaged_extended[substate] = []
        for i,(fr,metadata) in enumerate(zip(extended_substate['FR'],extended_substate['metadata'])):
            average_fr = np.nanmean(fr.values[:,(metadata.Region == stru) & (metadata.Type == types)],1)
            averaged_extended[substate].append(nts.Tsd(t = fr.times(),d = average_fr))
    return averaged_extended

def plot_fr_across_extended(extended:Dict[str,list],stru:str,types:str,ax):
    """
    Make figure with scatter of extended + average by same size group + delta

    Parameters
    ----------
    extended : Dict[list]
        given by :py:func:`merge_extended` after running :py:func:`process_all_sessions`
    """
    averaged_extended = average_extended(extended, stru, types)
    if ax is None:
        fig, ax = plt.subplots(3,2)
    plot_scatter_extended(averaged_extended['NREM'], colors['NREM'], ax=ax[0])
    plot_scatter_extended(averaged_extended['REM'], colors['REM'], ax=ax[1])
    plot_scatter_extended(averaged_extended['WAKE_HOMECAGE'], colors['WAKE_HOMECAGE'], ax=ax[2])
    # plot_delta(ax[6],colors)
    
    for i in ax:
        for j in i:
            plot.clean_axes(j)
    
    plt.tight_layout()
    plt.show()

    return ax

def plot_corr_fr_delta(df_firing_rates,df_delta_es,stru,ax = None):
    if ax is None:
        pass
    df = pd.merge(df_firing_rates,df_delta_es,'left',on = ['Rat','Day','Shank','Id','Region','Type','SessID'])
    df_stru_pyr = df[(df.Region == stru) & (df.Type == 'Pyr')]
    df_stru_pyr.dropna(inplace = True)

    states = ['NREM','REM','WAKE_HOMECAGE']

    for i,state in enumerate(states):
        x = np.linspace(-3,3,100)
        reg = linregress(df_stru_pyr['WAKE_HOMECAGE'],df_stru_pyr[f'delta_{state}'])
        y = reg.slope * x +reg.intercept
        ax[i].scatter(df_stru_pyr['WAKE_HOMECAGE'],df_stru_pyr[f'delta_{state}'],alpha = 0.3,s = 0.5, c = colors[state])
        ax[i].plot(10**x,y,color = colors[state])
        ax[i].text(-1,-0.5,f'r = {reg.rvalue:.4f}\np = {reg.pvalue:.4f}',transform = ax[i].transAxes)
        ax[i].semilogx()
        ax[i].set_ylim(-3,3)
        plot.clean_axes(ax[i])
        plot.forceAspect(ax[i])
if __name__ == '__main__':
    stru = 'BLA'

    plt.ion()
    extended = io.load_shelve('processed_data/binned_fr_extended')
    df_firing_rates =  pd.read_csv('processed_data/states_fr.csv') # generated by fr_states.py
    df_delta_es = pd.read_csv('processed_data/delta_extended.csv')
    df_delta_es.rename(columns={'NREM':'delta_NREM',
                        'REM':'delta_REM',
                        'WAKE_HOMECAGE':'delta_WAKE_HOMECAGE'},inplace=True)


    fig, ax = plt.subplots(3,3)
    plot_fr_across_extended(extended['merged_sessions'],stru,'Pyr',ax[:3,:2])
    plot_corr_fr_delta(df_firing_rates,df_delta_es,stru,ax[:,2])

    plt.savefig('plots/figures/extended.svg')
