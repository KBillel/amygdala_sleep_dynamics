import numpy as np
import pandas as pd
import scipy.stats

import seaborn as sns
import matplotlib.pyplot as plt

from bk import plot

from typing import Union, Optional,Tuple, Dict, Sequence

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
    print(reg)
    print(len(df_melt))      
    x = range(6000)
    y = x*reg.slope + reg.intercept 

    for i in df:
        sns.scatterplot(x = df.index/1_000_000,y = df[i],color = color,ax=ax[0],alpha = 0.8,s = 10)
    ax[0].plot(x,y,'k--')
    ax[0].text(4000,1.2,f'r = {reg.rvalue:.3f} \np = {reg.pvalue:.4f}')
    ax[0].set_ylim(-2,2)
    ax[0].set_xlim(-500,6000)
    plot.forceAspect(ax[0])
    ax[0].set_xlabel('Time from start of ext. sleep (s)')
    ax[0].set_ylabel('Firing rates (Z)')

    t,average,sem = average_by_equal_sized_group(df_melt,10)


    ax[1].errorbar(t/1_000_000,average,sem,linestyle='None',marker = '.',color = color,barsabove=True)
    ax[1]
    ax[1].plot(x,y,'k--')
    ax[1].set_ylim(-0.25,0.25)
    ax[1].set_xlim(0,4000)
    plot.forceAspect(ax[1])
    return False

def plot_delta(ax):
    """
    Plot barplot of delta FR beg vs end of extended sleep

    Parameters
    ----------
    ax : axis
        matplotlib axis
    """
    df = pd.read_csv('processed_data/fr.csv')
    sns.barplot(df[(df.Region == 'BLA') & (df.Type == 'Pyr')],order=['delta_NREM','delta_REM','delta_WAKE_HOMECAGE'],ax = ax)
    ax.set_ylim(-0.1,0.1)
    plot.forceAspect(ax)

def plot_fr_across_extended(extended:Dict[str,list]):
    """
    Make figure with scatter of extended + average by same size group + delta

    Parameters
    ----------
    extended : Dict[list]
        given by :py:func:`merge_extended` after running :py:func:`process_all_sessions`
    """
    fig, ax = plt.subplots(1, 7, figsize=(16, 8))
    plot_scatter_extended(extended['NREM'], 'grey', ax=ax[0:2])
    plot_scatter_extended(extended['REM'], 'orange', ax=ax[2:4])
    plot_scatter_extended(extended['WAKE_HOMECAGE'], 'green', ax=ax[4:6])
    plot_delta(ax[6])
    plt.tight_layout()
    plt.show()
    plt.savefig('plots/figures/decrease.svg')

if __name__ == '__main__':
    plt.style.use('ggplot')
