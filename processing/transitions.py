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
import shelve

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


def check_continuity(block, cont_th=1.5):
    cont_th_us = cont_th * 1_000_000
    starts = block['start'].values
    stops = block['end'].values
    delta_t = starts[1:] - stops[:-1]
    is_hole = np.any(delta_t > cont_th_us)
    return not is_hole


def find_transitions(states:Dict[str,nts.IntervalSet], n_states=2, min_durations:Dict[str,int]=None)->list[pd.DataFrame]:
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
    
    states = {name: intervals for name, intervals in states.items() if name != 'WAKE'}
    l_state_df = []
    for name, intervals in states.items():
        intervals['state'] = name
        l_state_df.append(intervals)
    state_df = pd.concat(l_state_df)
    state_df.sort_values(by='start', inplace=True)
    state_df.reset_index(drop = True, inplace=True)
    for irow, row in state_df.iterrows():
        if row['end'] - row['start'] < min_durations[row['state']]*1_000_000:
            state_df.loc[irow, 'state'] = 'HOLE'
    transitions = {}

    n_rows = len(state_df)
    for irow, row in state_df.iterrows():
        if irow > n_rows - n_states:
            break
        c_block = state_df.iloc[irow:irow+n_states]
        is_cont = check_continuity(c_block)
        if (not is_cont) or ('HOLE' in c_block['state'].values):
            continue
        trans_name = '-'.join(c_block['state'].values)
        prev_trans = transitions.get(trans_name, [])
        prev_trans.append(c_block)
        transitions[trans_name] = prev_trans
    return transitions

def compute_transitions_activity(neurons:ArrayLike,transitions:list[pd.DataFrame],nbins:Dict[str,int])->ArrayLike:
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
    activity = {}
    for tr_name,tr_l_intervals in transitions.items():
        fr_array_tr = []
        for interval in tr_l_intervals:
            
            l_spikes = []
            for irow,row in interval.iterrows():
                start_s = (row.start / 1_000_000) -0.5
                end_s = (row.end / 1_000_000) +0.5
                delta_t = (end_s - start_s) / nbins[row['state']]
                t,b = compute.binSpikes(neurons,start = start_s,stop = end_s,nbins = nbins[row['state']])
                b = b / delta_t
                l_spikes.append(b)
            fr_array = np.hstack(l_spikes)
            fr_array_tr.append(fr_array)
        activity[tr_name] = np.dstack(fr_array_tr)
    return activity



def plot_all_intervals(states):
    fig,ax = plt.subplots(2,1)
    plot.intervals(states['NREM'],'grey',ax =ax[0])
    plot.intervals(states['REM'],'orange',ax =ax[0])
    plot.intervals(states['WAKE_HOMECAGE'],'cyan',ax =ax[0])


def process_session(base_folder:Union[Path,str]= upath['base_folder'],
                    local_path:Union[Path,str]=upath['example_session'],
                    nbins=None,
                    min_durations=None,
                    save = False)->pd.DataFrame:
    
    md = load.session(base_folder,local_path)
    states = load.sleep_scoring(md)
    neurons,metadata = load.spikes(md)
    
    transitions = find_transitions(states,2,min_durations)
    activity = compute_transitions_activity(neurons[(metadata.Region == 'BLA') & (metadata.Type == 'Pyr')],transitions,nbins)

    if save: 
        with shelve.open(f'{md.get("session_name")}-transitions_activity') as f:
            f['metadata'] = metadata
            f['transitions'] = transitions
            f['activity'] = activity
    return transitions,activity

def process_all_sessions(base_folder:Union[Path,str]= upath['base_folder'],**kwargs)->Tuple:
    """
    Run :py:func:'process_session' for all session in the dataset

    Parameters
    ----------
    base_folder : Union[Path,str], optional
        _description_, by default upath['base_folder']

    Returns
    -------
    _type_
        _description_
    """

    session_list = load.session_list()
    all_df = []
    all_extended_fr = []
    for p in tqdm(session_list.Path):
        try:
            print(p)
            df = process_session(local_path=p,**kwargs)
            all_df.append(df)
        except:
            print(f'{p} not taken care of because bug')
    

    return all_df

if __name__ == '__main__':

    min_durations = {
        'NREM':200,
        'REM':50,
        'WAKE_HOMECAGE':200,
        'DROWSY': 25
    }

    nbins = {
        'NREM':30,
        'REM':12,
        'WAKE_HOMECAGE':30,
        'DROWSY':1}
    
    save = True
    
    process_session(nbins = nbins, min_durations=min_durations,save = save)

