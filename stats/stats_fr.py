import pandas as pd

import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.sandbox as sms

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
if __name__ == '__main__':
    stru = 'BLA'
    
    df_firing_rates = pd.read_csv('processed_data/states_fr.csv')
    df_firing_rates['UUID'] = df_firing_rates.apply(lambda row:f'{row.Rat:02d}{row.Day:02d}{row.Shank:02d}{row.Id:02d}',axis = 1)
    df_firing_rates_stru = df_firing_rates[(df_firing_rates.Region == stru) & ((df_firing_rates.Type == 'Pyr') )]
    df_melt = df_firing_rates_stru.melt(id_vars=['Rat','UUID','Region','Type'],value_vars=['NREM','REM','WAKE_HOMECAGE'],
                                   var_name='State',value_name='FR')

    values_to_test = [df_firing_rates_stru[state] for state in ['NREM','REM','WAKE_HOMECAGE']]
    scipy.stats.kruskal(*values_to_test)

    mc = sms.stats.multicomp.MultiComparison(df_melt.FR,df_melt.State)
    all_test = mc.allpairtest(scipy.stats.wilcoxon)
    print(all_test[0])
    print(all_test[1])