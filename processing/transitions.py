from settings import upath

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from bk import load
from bk import stats
from bk import compute
from bk import plot

import neuroseries as nts

from tqdm import tqdm
from functools import reduce

from pathlib import Path
from typing import Union, Optional,Tuple, Dict, Sequence
from numpy.typing import ArrayLike


def in_state(t:int,states:Dict[str,nts.IntervalSet])->set:
    """
    Return what intervals of states t is inside. 

    Parameters
    ----------
    t : int
        time to test
    states : Dict[str,nts.Intervalset]
        states as given by :py:func:`load.intervals`

    Returns
    -------
    set(l_states)
        set of states in which t fits
    """
    t = nts.Ts(t,dtype = 'float64')
    l_states = []
    for state,intervals in states.items():
        if np.all(np.isnan(intervals.in_interval(t))):
            continue
        else:
            l_states.append(state)
    return set(l_states)

def closest_interval(state:nts.IntervalSet,interval:nts.IntervalSet)->Tuple[nts.IntervalSet]:
    """
    Return closest left and right interval in state of state

    Parameters
    ----------
    state : nts.IntervalSet
        state to look for interval
    interval : nts.IntervalSet
        interval to look left and right

    Returns
    -------
    Tuple[nts.IntervalSet]
        closest left and right intervals. 
    """
    idx_before = np.argmin(np.abs(state.end - interval.start))
    idx_after = np.argmin(np.abs(state.start - interval.end))

    return state.iloc[idx_before],state.iloc[idx_after]

def find_transitions(states:Dict[str,nts.IntervalSet],previous_state:str='NREM',state:str='REM',next_state:str=None,min_durations:Dict[str,int]=None,epsilon:float = 1.5)->list[pd.DataFrame]:
    """
    This function compute timing of transitions from a state to another. 

    Parameters
    ----------
    states : Dict[str,nts.Intervalset]
        states as given by :py:func:`load.intervals`
    previous_state : str, optional
        previous_state to look for, by default 'NREM'
    state : str, optional
        state to which the transitions is occuring, by default 'REM'
    next_state : str, optional
        following state,, by default None
    epsilon : float, optional
        next or previous state should be at most epsilon after the end of state, by default 1.5

    Returns
    -------
    list[pd.Dataframe]
        List of dataframe. Each DataFrame contains a transition event
    """
    
    epsilon *= 1_000_000 # conversion to µs

    for s,value in min_durations.items():
        states[s] = states[s].drop_short_intervals(value,time_units='s').reset_index(drop = True)
        if len(states[s]) == 0:
            print(f'There is no {s} longer than {min_durations[s]} in states')
            return []

    
    if previous_state is not None:
        t_before = states[state].start - epsilon
        states_before = [in_state(t,states) for t in t_before]

    if next_state is not None:
        states_names = [previous_state,state,next_state]

        t_after = states[state].end + epsilon
        states_after = [in_state(t,states) for t in t_after]
        previous_valid = [previous_state in s for s in states_before]
        next_valid = [next_state in s for s in states_after]
        
        valid =  [a and b for a,b in zip(previous_valid,next_valid)]
        
    else:
        states_names = [previous_state,state]
        valid = [previous_state in s for s in states_before]
    
    transitions = []
    for interval in states[state][valid].iloc:
        previous_interval = closest_interval(states[previous_state],interval)[0]
        if next_state is not None:
            next_interval = closest_interval(states[next_state],interval)[1]
            df_ = pd.DataFrame([previous_interval,interval,next_interval])
        else:
            df_ = pd.DataFrame([previous_interval,interval])
            
        # adding, removing 0.5s on each start/end because states are not contigus other wise
        df_.start = df_.start - 500_000 
        df_.end = df_.end + 500_000
        df_['state'] = states_names
        transitions.append(df_.reset_index(drop=True))
        
    return transitions

def compute_transition_activity(neurons:ArrayLike,l_transitions_intervals:list[pd.DataFrame],nbins:Dict[str,int])->ArrayLike:
    """
    Function compute the normalized activity for each neurons for all transitions

    Parameters
    ----------
    neurons : ArrayLike
        list of neurons given by :py:func:`load.spikes`
    l_transitions_intervals : list[pd.DataFrame]
        list of transitions intervals as given by :py:func`find_transitions`
    nbins : Dict[str,int]
        number of bin used for each state

    Returns
    -------
    ArrayLike
        _description_
    """

    for transition_interval in l_transitions_intervals:
        pass


def process_session(base_folder:Union[Path,str]= upath['base_folder'],local_path:Union[Path,str]=upath['example_session'],discarded_states:Sequence[str] = ('DROWSY','WAKE'))->pd.DataFrame:
    md = load.session(base_folder=base_folder,local_path=local_path)
    discarded_states = set(discarded_states)

    neurons,metadata = load.spikes(md)
    metadata['SessID'] = metadata.index

    id_columns = list(metadata.columns)
    states = load.sleep_scoring(md,discard = discarded_states)


def plot_all_intervals(states):
    fig,ax = plt.subplots(2,1)
    plot.intervals(states['NREM'],'grey',ax =ax[0])
    plot.intervals(states['REM'],'orange',ax =ax[0])
    plot.intervals(states['WAKE_HOMECAGE'],'cyan',ax =ax[0])


if __name__ == '__main__':
    md = load.session()
    discarded_states = set()
    states = load.sleep_scoring(md,discard = discarded_states)

    min_durations = {
        'NREM':200,
        'REM':50,
        'WAKE_HOMECAGE':200
    }

    transitions = find_transitions(states,'NREM','REM',min_durations = min_durations)

    fig,ax = plt.subplots(2,1,sharex=True)
    plot.intervals(states['NREM'],'grey',ax =ax[0])
    plot.intervals(states['REM'],'orange',ax =ax[0])
    plot.intervals(states['WAKE_HOMECAGE'],'cyan',ax =ax[0])

    for t in transitions:
        plot.intervals(t,'red',ax=ax[1])
    
    plt.show()





