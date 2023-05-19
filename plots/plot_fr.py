import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from bk import plot
from typing import Union, Optional,Tuple, Dict, Sequence

from statannotations.Annotator import Annotator
from settings import colors
import itertools
from scipy.stats import linregress

def set_rem_labels(df):
    rem_on = df.pInc <= 0.001
    rem_off = df.pDec <= 0.001
    rem_unknown = ((df.pInc > 0.001) & (df.pDec > 0.001))

    df['REM_label'] = np.nan
    df['REM_label'][rem_on] = 'REM_ON'
    df['REM_label'][rem_off] = 'REM_OFF'
    df['REM_label'][rem_unknown] = 'Unknown'

    return df


def boxenplot_firing_rates(df,stru,ax = None):

    if ax is None:
        fig,ax = plt.subplots()

    states = ['REM','NREM','WAKE_HOMECAGE']
    pairs = [[('Pyr', 'NREM'), ('Pyr', 'REM')],
             [('Pyr', 'NREM'), ('Pyr', 'WAKE_HOMECAGE')],
             [('Pyr', 'REM'), ('Pyr', 'WAKE_HOMECAGE')],

             [('Int', 'NREM'), ('Int', 'REM')],
             [('Int', 'NREM'), ('Int', 'WAKE_HOMECAGE')],
             [('Int', 'REM'), ('Int', 'WAKE_HOMECAGE')]]

    df_stru = df[(df.Region == stru)]

    df_melt = pd.melt(df_stru,value_vars=states,value_name='FR',id_vars = 'Type',var_name='States')
    df_melt['FR'] = np.log10(df_melt['FR'])
    df_melt['FR'].replace([np.inf,-np.inf],np.nan,inplace=True)
    df_melt.dropna(inplace=True)

    
        
    
    ### Violin plots
    ax.set_title('Violin Plots')
    plotting_params = {'data':df_melt,
                       'x':'Type',
                       'y':'FR',
                       'hue':'States',
                       'hue_order':['NREM','REM','WAKE_HOMECAGE'],
                       'palette':[colors['NREM'], colors['REM'], colors['WAKE_HOMECAGE']],
                       'ax':ax}

    sns.boxenplot(**plotting_params)
    annotator = Annotator(pairs = pairs,**plotting_params)
    stats = annotator.configure(test="Wilcoxon",comparisons_correction = 'Bonferroni').apply_and_annotate()
    ax.set_xlabel('States')
    ax.set_ylabel('FiringRates')
    y_ticks = np.arange(-3,4)
    ax.set_yticks(y_ticks,pow(10.0,y_ticks))
    ax.legend(loc = 'lower right')

    return ax,(pairs,stats)

def cumsum_curves_firing_rates(df,stru,states_name,ax = None):
    df_stru = df[(df.Region == stru)]
    df_pyr= df_stru[(df.Type == 'Pyr')]
    df_int = df_stru[(df.Type == 'Int')]
    
    
    if ax is None:
        fig,ax = plt.subplots(1,2)
    for state in states_name:
        fr_pyr = df_pyr[state]
        fr_int = df_int[state]
        ax[0].plot(np.sort(fr_pyr),np.arange(len(fr_pyr))/len(fr_pyr),c = colors[state])
        ax[1].plot(np.sort(fr_int),np.arange(len(fr_int))/len(fr_int),c = colors[state])
    
    ax[0].semilogx()
    ax[0].set_xlim(-3,3)
    ax[0].set_xlabel('Firing Rates (Hz)')
    ax[0].set_ylabel('Proportion')
    ax[0].set_title('Principal Neurons')
    
    ax[1].set_xlim(0,fr_int.max())
    ax[1].set_xlabel('Firing Rates (Hz)')
    ax[1].set_ylabel('Proportion')
    ax[1].set_title('Interneurons')

    
def proportion_rem_on(df,stru,ax = None):
    if ax is None:
        fig,ax = plt.subplots()
    df = df[(df.Region == stru)]
    
    data = df.groupby(['Region','Type'])['REM_label'].value_counts(normalize = True).unstack()
    data = data[['REM_ON','REM_OFF','Unknown']]
    data = data.loc['BLA']
    print(data)
    bar = data.plot.bar(stacked = True,ax = ax,color = colors)

def corr_rem_nrem_fr(df,stru,ax = None):
    quantile_labels = ['VL','L','M','H','VH']

    if ax is None:
        fig,ax = plt.subplots(1,2)
    
    df['ratio'] = (df['REM']-df['NREM']) / (df['NREM'] + df['REM'])
    df = df.replace([np.inf,-np.inf],np.nan,)
    df = df.dropna()
    
    df_stru_pyr = df[(df.Region == stru) & (df.Type == 'Pyr')]
    df_stru_int = df[(df.Region == stru) & (df.Type == 'Int')]

    reg_pyr = linregress(df_stru_pyr['WAKE_HOMECAGE'],df_stru_pyr['ratio'])
    reg_int = linregress(df_stru_int['WAKE_HOMECAGE'],df_stru_int['ratio'])
    

    #print reg and scatter for pyr
    x = np.linspace(-3,3,100)
    ax[0].scatter(df_stru_pyr['WAKE_HOMECAGE'],df_stru_pyr['ratio'],s = 1,c = 'g')
    y = reg_pyr.slope * x + reg_pyr.intercept
    ax[0].plot(10**x,y,'g')

    ax[0].scatter(df_stru_int['WAKE_HOMECAGE'],df_stru_int['ratio'],s = 1,c = 'k')
    y = reg_int.slope * x + reg_int.intercept
    ax[0].plot(10**x,y,'k')

    ax[0].semilogx()
    ax[0].set_ylim(-1,1)

    ax[0].set_xlabel('Firing Rates (WAKE_HOMECAGE)')
    ax[0].set_ylabel(r'Ratio $\frac{REM-NREM}{REM+NREM}$')
    # plot.forceAspect(ax[0]) 
    


    #Make quintiles
    plotting_params = {'data':df_stru_pyr,
              'x':'Quantiles',
              'y':'ratio',
              'palette':'Greens',
              'ax':ax[1]}
    
    df_stru_pyr['Quantiles'] = pd.qcut(df_stru_pyr['WAKE_HOMECAGE'],5,quantile_labels)
    pairs = list(itertools.combinations(df_stru_pyr.Quantiles.unique(),2))
    
    sns.boxplot(**plotting_params)
    annotator = Annotator(pairs=pairs,**plotting_params)
    _,stats = annotator.configure(test="Mann-Whitney",comparisons_correction = 'Bonferroni').apply_and_annotate()
    ax[1].set_ylim(-1,3.5)


if __name__ == '__main__':
    plt.ion()
    # Load Data
    df = pd.read_csv('processed_data/states_fr.csv') # generated by fr_states.py
    rem_on_off = pd.read_csv('processed_data/rem_on.csv')
    rem_on_off = set_rem_labels(rem_on_off)
    
    states_name = ['NREM','REM','WAKE_HOMECAGE']


    # Plots
    fig,ax = plt.subplot_mosaic("""AAAAADDD
                                   BBCCEEFF""",figsize = (12,8),layout="tight")
    
    
    _,(pairs,stats) = boxenplot_firing_rates(df,'BLA',ax = ax['A'])
    cumsum_curves_firing_rates(df,'BLA',states_name = states_name,ax = (ax['B'],ax['C']))
    proportion_rem_on(rem_on_off,'BLA',ax['D'])
    corr_rem_nrem_fr(df,'BLA',(ax['E'],ax['F']))
    for _,a in ax.items(): plot.clean_axes(a)
    # plt.tight_layout()
    fig.savefig('output.png')
    fig.savefig('plots/figures/fr.svg')
    plt.close(fig)