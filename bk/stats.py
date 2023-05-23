import numpy as np
import bk.compute
import bk.multi
import scipy.stats as ss
import neuroseries as nts
import pandas as pd

# Remi imports :
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Sequence,List
from numpy.typing import ArrayLike
from settings import upath


from serde import serde
from serde.json import to_json

from dataclasses import dataclass

def poisson(baseRate:float, counts:int, time:float) -> Tuple[float, float, float]:
    """
    Compute the poisson test to see of the number of observations during an elapsed time 
    is hight than expected by the poisson distribution at baseRate

    Parameters
    ----------
    baseRate : float
        baseRate of the neuron in the off state
    counts : int
        number of observation
    time : float
        elapse time

    Returns
    -------
    Tuple[float, float, float]
        pInc : float
            probability of increased observation expecteded
        pDec : float
            probability of decreased observation expecteded
        suprise : float
            log ratio of pDec / pInc
    """
    eps = np.spacing(1)

    lam = baseRate*time

    pInc = 1 - ss.poisson.cdf(counts-1, lam)
    pDec = ss.poisson.cdf(counts, lam)
    surprise = np.log((pDec+eps)/(pInc + eps))
    return pInc, pDec, surprise


def poisson_test(neurons:ArrayLike, on_state:nts.IntervalSet, off_state:nts.IntervalSet)-> pd.DataFrame:
    """
    Compute the poisson test for array of neurons

    Parameters
    ----------
    neuron : ArrayLike
        Array of neurons as Tsd format
    on_state : nts.IntervalSet
        On state (ripples, rem etc ...)
    off_state : nts.IntervalSet
        off state (off ripples, NREM etc ...)

    Returns
    -------
    pd.DataFrame
        Dataframe with the pValue of increase/decrease and the surprise value of the test
    """
    pInc = np.zeros(len(neurons)) + np.nan
    pDec = np.zeros(len(neurons)) + np.nan
    surprise = np.zeros(len(neurons)) + np.nan

    for i, n in enumerate(neurons):
        baseRate = len(n.restrict(off_state)) / \
            off_state.tot_length(time_units='s')
        count = len(n.restrict(on_state))
        time = on_state.tot_length(time_units='s')
        pInc[i], pDec[i], surprise[i] = poisson(baseRate, count, time)

    stats = pd.DataFrame(np.array([pInc, pDec, surprise]).T, columns=[
                         'pInc', 'pDec', 'surprise'])
    return stats


def rayleigh(phases, weights=None):
    r = bk.compute.mean_resultant_length(phases, weights)
    if weights is not None:
        n = np.sum(weights)
    else:
        n = len(phases)

    R = r*n
    z = R**2 / n

    pvalue = np.exp(np.sqrt(1+4*n+4*(n**2 - R**2))-(1+2*n))
    return pvalue


def ppc(neuron, phases, jitter_max, n_spikes, n_shuffles, n_workers):
    return None


def shuffles_pvalue(shuffle, value):
    return np.min((len(shuffle[shuffle > value])/len(shuffle), len(shuffle[shuffle < value])/len(shuffle)))


def formatting_pvalues(pvalues):

    one_star = pvalues < 0.05
    two_star = pvalues < 0.01
    three_star = pvalues < 0.001

    no_star = pvalues >= 0.05
    pvalues = pvalues.astype('object')

    pvalues[one_star] = '*'
    pvalues[two_star] = '**'
    pvalues[three_star] = '***'
    pvalues[no_star] = 'N.S'

    return pvalues



def from_scipy(stats_data,panel_name,n):
    stat = Stats(panel_name=panel_name,
                test=str(stats_data.__class__),
                n = n,
                pvalue=stats_data.pvalue,
                stat_value=stats_data.statistic,
                name = '')
    return List_Stats([stat])


def from_statannon(stats_data,panel_name):
    l_stats = []
    for stat in stats_data:
        l_stats.append(Stats(panel_name=panel_name,
                             test = stat.data.test_description,
                             n = [len(group['group_data']) for group in stat.structs],
                             pvalue = stat.data.pvalue,
                             stat_value = stat.data.stat_value,
                             name = f'{stat.data.group1} vs {stat.data.group2}'))

    l_stats = List_Stats(l_stats)
    return l_stats


@serde
@dataclass
class Stats:
    panel_name: str
    test: str
    n: List[int]
    pvalue : float
    stat_value: float
    name: str
    n_total: Optional[int] = None
    def __post_init__(self):
        self.n_total = sum(self.n)

@serde
@dataclass
class List_Stats:
    l_stats: List[Stats]
    def __getitem__(self,idx):
        return self.l_stats[idx]
    def append(self,item:Stats):
        self.l_stats.append(item)
    def extend(self,items):
        self.l_stats.extend(items)
    def save(self,path:str):
        with open(path,'w') as jf:
            jf.write(to_json(self,indent = 4))
    