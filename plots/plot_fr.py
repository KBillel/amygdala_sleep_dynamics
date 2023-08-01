import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from scipy.stats import linregress, kruskal
from statannotations.Annotator import Annotator

from bk import io
from bk import plot
from bk.stats import from_statannon, from_scipy
from plots.plot_transitions import plot_activity_at_transitions
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


def boxenplot_firing_rates(df,stru,ax = None):

    if ax is None:
        fig,ax = plt.subplots()

    #FIXME States should be a params like in cumsum_curves_firing_rates
    states = ['REM','NREM','WAKE_HOMECAGE']
    pairs = [[('Pyr', 'NREM'), ('Pyr', 'REM')],
             [('Pyr', 'NREM'), ('Pyr', 'WAKE_HOMECAGE')],
             [('Pyr', 'REM'), ('Pyr', 'WAKE_HOMECAGE')],

             [('Int', 'NREM'), ('Int', 'REM')],
             [('Int', 'NREM'), ('Int', 'WAKE_HOMECAGE')],
             [('Int', 'REM'), ('Int', 'WAKE_HOMECAGE')]]

    df_stru = df[(df.Region == stru)]

    df_melt = pd.melt(df_stru,value_vars=states,value_name='FR',id_vars = 'Type',var_name='States')
    print('\n\n\n****FIRING RATES****\n\n\n')
    print(df_melt.groupby(['States','Type']).mean())
    print(df_melt.groupby(['States','Type']).std())
    print('\n\n\n****FIRING RATES****\n\n\n')

    # df_melt['FR'] = np.log10(df_melt['FR'])
    df_melt['FR'].replace([np.inf,-np.inf],np.nan,inplace=True)
    df_melt.dropna(inplace=True)

    ### Violin plots
    plotting_params = {'data':df_melt,
                       'x':'Type',
                       'y':'FR',
                       'hue':'States',
                       'hue_order':['NREM','REM','WAKE_HOMECAGE'],
                       'palette':[colors['NREM'], colors['REM'], colors['WAKE_HOMECAGE']],
                       'ax':ax}
    ax.semilogy()

    sns.boxenplot(**plotting_params)
    annotator = Annotator(pairs = pairs,**plotting_params)
    _,stats_data = annotator.configure(test="Wilcoxon",comparisons_correction = 'Bonferroni').apply_and_annotate()
    formatted_stats = from_statannon(stats_data,'FR_States')
    ax.set_xlabel('States')
    ax.set_ylabel('Firing Rates\n(Hz)')
    # y_ticks = np.arange(-3,4)
    # tick_label = pow(10.0,y_ticks)
    # ax.set_yticks(y_ticks,[f'$10^{{{int(y)}}}$' for y in y_ticks])
    # ax.legend(loc = 'lower right')

    mask_int = df_stru['Type'] == 'Pyr'
    print('ICI')
    print(scipy.stats.mannwhitneyu(df_stru[mask_int]['NREM'],df_stru[mask_int]['REM']))
    print(scipy.stats.mannwhitneyu(np.log10(df_stru[mask_int]['NREM']),np.log10(df_stru[mask_int]['REM'])))

    return ax,formatted_stats


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

def plot_histograms_firing_rates(df,stru,state,ax = None):
    if ax is None:
        fig,ax = plt.subplots(1,1)
    df_stru = df[(df.Region == stru)]
    df_pyr= df_stru[(df.Type == 'Pyr')]
    bins = np.logspace(np.log10(0.001), np.log10(100) , 50)
    c,bins_cut = pd.qcut(df_pyr[state],5,retbins=True)
    for i in bins_cut[1:-1]:
        ax.axvline(i,c = 'r')
    ax.hist(df_pyr[state],bins = bins,color = colors[state])
    ax.semilogx()    
    ax.set_xlabel(f'Firing Rates\n (Hz - {state})')
    ax.set_ylabel('Counts')

    
def proportion_rem_on(df,stru,ax = None):
    if ax is None:
        fig,ax = plt.subplots()
    df = df[(df.Region == stru)]
    
    data = df.groupby(['Region','Type'])['REM_label'].value_counts(normalize = True).unstack()
    data = data[['REM_ON','REM_OFF','Unknown']]
    data = data.loc['BLA']
    print(data)
    bar = data.plot.bar(stacked = True,ax = ax,color = colors)
    ax.set_ylabel('Proportion')


def corr_rem_nrem_fr(df,stru,quintiles_state,ax = None):
    quantile_labels = ['VL','L','M','H','VH']

    if ax is None:
        fig,ax = plt.subplots(1,2)
    
    df['ratio'] = (df['REM']-df['NREM']) / (df['NREM'] + df['REM'])
    df = df.replace([np.inf,-np.inf],np.nan,)
    df = df.dropna()
    
    df_stru_pyr = df[(df.Region == stru) & (df.Type == 'Pyr')]
    df_stru_int = df[(df.Region == stru) & (df.Type == 'Int')]

    reg_pyr = linregress(df_stru_pyr[quintiles_state],df_stru_pyr['ratio'])
    reg_int = linregress(df_stru_int[quintiles_state],df_stru_int['ratio'])
    print(reg_int)
    

    #print reg and scatter for pyr
    x = np.linspace(-3,3,100)
    ax[0].scatter(df_stru_pyr[quintiles_state],df_stru_pyr['ratio'],s = 1,c = 'g')


    y = reg_pyr.slope * x + reg_pyr.intercept
    ax[0].plot(10**x,y,'g')
    ax[0].scatter(df_stru_int[quintiles_state],df_stru_int['ratio'],s = 1,c = 'k')
    y = reg_int.slope * x + reg_int.intercept
    ax[0].plot(10**x,y,'k')


    ax[0].text(0.2,1,f'r = {reg_pyr.rvalue:.3g}\np = {reg_pyr.pvalue:.3g}',transform = ax[0].transAxes,c = 'g')
    ax[0].text(0.6,1,f'r = {reg_int.rvalue:.3g}\np = {reg_int.pvalue:.3g}',transform = ax[0].transAxes,c = 'k')
    ax[0].semilogx()
    ax[0].set_ylim(-1,1)

    ax[0].set_xlabel(f'Firing Rates\n(Hz - {quintiles_state})')
    ax[0].set_ylabel(r'Ratio $\frac{REM-NREM}{REM+NREM}$')
    # plot.forceAspect(ax[0]) 
    #Make quintiles

    
    df_stru_pyr['Quantiles'] = pd.qcut(df_stru_pyr[quintiles_state],5,quantile_labels)
    
    plotting_params = {'data':df_stru_pyr,
            'x':'Quantiles',
            'y':'ratio',
            'palette':'Greens',
            'ax':ax[1]}
    kruskal_stats = kruskal(*[df_stru_pyr[df_stru_pyr['Quantiles'] == q].ratio for q in quantile_labels])
    kruskal_stats = from_scipy(kruskal_stats,'FR_States',df_stru_pyr.groupby('Quantiles').count().Rat.to_list())
    pairs = list(itertools.combinations(df_stru_pyr.Quantiles.unique(),2))
    
    sns.boxplot(**plotting_params)
    annotator = Annotator(pairs=pairs,**plotting_params)
    _,stats_data = annotator.configure(test="Mann-Whitney",comparisons_correction = 'Bonferroni').apply_and_annotate()
    mann_whitney = from_statannon(stats_data,'Corr_IncreaseREM_FRWake')

    kruskal_stats.extend(mann_whitney)
    ax[1].set_ylim(-1,3.5)
    ax[1].set_ylabel(r'Ratio $\frac{REM-NREM}{REM+NREM}$')

    return ax,kruskal_stats


def plot_transitions_panel(transitions, df_firing_rates, stru,norm,state, params, transition_name, ax):
    states = transition_name.split('-')
    bin_state = [(s,params['nbins'][s]) for s in states]
    
    c_transitions = transitions[transition_name]
    c_activity = c_transitions['activity']
    c_metadata = pd.merge(c_transitions['metadata'],df_firing_rates,on = ['Rat','Day','Shank','Id','Region','Type'],how='left')
    plot_activity_at_transitions(c_activity,c_metadata,stru,norm = norm,quantile=state,ax=ax,bin_state = bin_state)
    # plot_activity_at_transitions(c_activity,c_metadata,stru,norm = 'zscore',quantile=state,ax=ax[1],bin_state = bin_state)
    ax.set_xlabel('Times (bins)')
    if norm == 'zscore':
        ax.set_ylabel('Firing Rates\n(zscore)')
        ax.set_ylim(-1.5,1.5)
    else:
        ax.set_ylabel('Firing Rates\n(Hz)')
        ax.semilogy()


    ax.set_title(transition_name)
    # ax[1].set_xlabel('Times (bins)')


if __name__ == '__main__':
    plt.ion()
    # Load Data
    stru = 'BLA'
    quantile_state = 'WAKE_HOMECAGE'

    df = pd.read_csv('processed_data/states_fr.csv') # generated by fr_states.py
    rem_on_off = pd.read_csv('processed_data/rem_on.csv')
    rem_on_off = set_rem_labels(rem_on_off)
    
    transitions = io.load_shelve('processed_data/transitions')['merged_sessions']
    with open('processed_data/transitions.json','r') as jf:
        params = json.load(jf)

    states_name = ['NREM','REM','WAKE_HOMECAGE']
    transitions_of_interest = {'NREM-REM':[0,30]}

    # Plots
    fig,ax = plt.subplot_mosaic("""AABBCCII
                                   AABBCCII
                                   AABBCCII
                                   DDFGGGHH
                                   DDFGGGHH
                                   EEFGGGHH
                                   EEFGGGHH
                                   """,
                                   figsize = (14,8),
                                   layout='tight')
    

    ax_corr,stats_boxplot = boxenplot_firing_rates(df,'BLA',ax = ax['A'])
    cumsum_curves_firing_rates(df,'BLA',states_name = states_name,ax = (ax['B'],ax['C']))
    plot_histograms_firing_rates(df,'BLA',quantile_state,ax = ax['I'])
    proportion_rem_on(rem_on_off,'BLA',ax['F'])
    ax_box,stats_corr = corr_rem_nrem_fr(df,'BLA',quantile_state,(ax['G'],ax['H']))

    plot_transitions_panel(transitions,df,stru,None,None,params,'NREM-REM',ax['D'])
    plot_transitions_panel(transitions,df,stru,'zscore',quantile_state,params,'NREM-REM',ax['E'])
    
    for _,a in ax.items(): plot.clean_axes(a)
    # plt.tight_layout()
    fig.savefig('plots/figures/supp-fr.png')
    fig.savefig('plots/figures/supp-fr.svg')

    stats_boxplot.extend(stats_corr)
    stats_boxplot.save('plots/figures/fr.json')


     ## Stats :
