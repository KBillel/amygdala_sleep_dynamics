from settings import upath, states_nbins, min_durations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from bk import load
from bk import stats
from bk import compute
from bk import plot
from bk import io

import neuroseries as nts

from tqdm import tqdm
from functools import reduce
import shelve

from pathlib import Path
from typing import Union, Optional, Tuple, Dict, List
from numpy.typing import ArrayLike



def check_continuity(block: nts.IntervalSet, cont_th: float = 1.5) -> bool:
    """
    Check if there are holes longer than cont_th in the intervals.

    Parameters
    ----------
    block : nts.IntervalSet
        block of interval set to be tested
    cont_th : float, optional
        , by default 1.5

    Returns
    -------
    not is_hole : bool
        Return True if not gap in the intervals
    """
    cont_th_us = cont_th * 1_000_000
    starts = block['start'].values
    stops = block['end'].values
    delta_t = starts[1:] - stops[:-1]
    is_hole = np.any(delta_t > cont_th_us)
    return not is_hole


def make_contigous(transition:pd.DataFrame)->pd.DataFrame:
    """
    Force interval DataFrame to become contigous so the end of each epoch match the start of the next

    Parameters
    ----------
    transition : pd.DataFrame
        c_block of find_transitions
    Returns
    -------
    pd.DataFrame
        contigous interval dataframe
    """
    s_0 = transition.start.values
    e_0 = transition.end.values
    state = transition.state.values

    s_p1 = np.roll(s_0,-1)
    s_p1[-1] = e_0[-1]

    e_m1 = np.roll(e_0,1)
    e_m1[0] = s_0[0]

    s = (s_0 + e_m1) / 2
    e = (e_0 + s_p1) / 2

    df = {'start':s.astype(int),
          'end':e.astype(int),
          'state':state}

    return pd.DataFrame(df)


def find_transitions(states: Dict[str, nts.IntervalSet],
                     n_states: int = 2, 
                     min_durations: Dict[str, int] = None,
                     contigous:bool = True) -> dict:
    """
    Find all transitions in states

    Parameters
    ----------
    states : Dict[str, nts.IntervalSet]
        states as given by :py:func:`load.sleep_scoring`
    n_states : int, optional
        _description_, by default 2
    min_durations : Dict[str, int], optional
        _description_, by default None
    contigous : bool, optional
        _description_, by default True

    Returns
    -------

    """
    states = {name: intervals for name,
              intervals in states.items() if name != 'WAKE'}
    l_state_df = []
    for name, intervals in states.items():
        intervals = intervals.copy()
        intervals['state'] = name
        l_state_df.append(intervals)
    state_df = pd.concat(l_state_df)
    state_df.sort_values(by='start', inplace=True)
    state_df.reset_index(drop=True, inplace=True)
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
        if contigous:
            c_block = make_contigous(c_block)
        trans_name = '-'.join(c_block['state'].values)
        prev_trans = transitions.get(trans_name, [])
        prev_trans.append(c_block)
        transitions[trans_name] = prev_trans
    return transitions


def compute_transitions_activity(neurons: ArrayLike, 
                                 transitions: Dict[str, List[nts.IntervalSet]],
                                 nbins: Dict[str, int]) -> ArrayLike:
    """
    Function compute the normalized activity for each neurons for all transitions

    Parameters
    ----------
    neurons : ArrayLike
        list of neurons given by :py:func:`load.spikes`
    transitions : Dict[List[nts.IntervalSet]]

    nbins : Dict[str,int]
        number of bin used for each state

    Returns
    -------
    ArrayLike
        _description_
    """
    activity = {}
    for tr_name, tr_l_intervals in transitions.items():
        fr_array_tr = []
        for interval in tr_l_intervals:
            l_spikes = []
            for irow, row in interval.iterrows():
                start_s = (row.start / 1_000_000)
                end_s = (row.end / 1_000_000)
                delta_t = (end_s - start_s) / nbins[row['state']]
                t, b = compute.binSpikes(neurons, start=start_s, stop=end_s, nbins=nbins[row['state']])
                b = b / delta_t
                l_spikes.append(b)
            fr_array = np.hstack(l_spikes)
            fr_array_tr.append(fr_array)
        activity[tr_name] = np.dstack(fr_array_tr)
    return activity


def save_data(session: Dict, 
              metadata: pd.DataFrame, 
              transitions: Dict[str, nts.IntervalSet], 
              activity: Dict[str, ArrayLike],
              nbins: Dict[str, int], 
              min_durations: Dict[str, int]) -> None:
    """
    Save the data output by :py:func:'process_session' in a shelve

    Parameters
    ----------
    session : Dict
    metadata : pd.DataFrame
    transitions : Dict[str, nts.IntervalSet]
    activity : Dict[str, ArrayLike]
    nbins : Dict[str, int]
    min_durations : Dict[str, int]
    """
    
    params = {'nbins': nbins,
              'min_durations': min_durations}

    d = {'unique_sessions': {session['session_name']: {'session':session,
                                                       'metadata': metadata,
                                                       'transitions': transitions,
                                                       'activity': activity,
                                                       'params': params}}}

    io.save_shelve('processed_data/transitions', dict=d, params=params)


def process_session(base_folder: Union[Path, str] = upath['base_folder'],
                    local_path: Union[Path, str] = upath['example_session'],
                    nbins: Dict[str, int] = None,
                    min_durations: Dict[str, int] = None,
                    save: bool = False,
                    force: bool = False) -> pd.DataFrame:
    """
    Process a session with computation relative to firing rates at transitions


    Parameters
    ----------
    base_folder : Union[Path,str], optional
        Path to the dataset, by default upath['base_folder']
    local_path : Union[Path,str], optional
        Relative path to the session, by default upath['example_session']
    nbins : Dict[str,int], optional
        Number of bins to divide each epochs, by default None
    min_durations : _type_, optional
        Minimum duration of each states to integrated in the analysis, by default None
    save : bool, optional
        by default False

    Returns
    -------
    pd.DataFrame
        _description_
    """
    session = load.session(base_folder, local_path)
    if not force:
        data = io.load_shelve('processed_data/transitions')
        if ('unique_sessions' in data) and (session['session_name'] in data['unique_sessions']):
            c_data = data['unique_sessions'][session['session_name']]
            return c_data['session'], c_data['metadata'], c_data['transitions'], c_data['activity']

    states = load.sleep_scoring(session)
    neurons, metadata = load.spikes(session)

    transitions = find_transitions(states, n_states=1, min_durations=min_durations)
    transitions.update(find_transitions(states, n_states=2, min_durations=min_durations))
    transitions.update(find_transitions(states, n_states=3, min_durations=min_durations))
    activity = compute_transitions_activity(neurons, transitions, nbins)

    if save:
        save_data(session,metadata, transitions, activity,nbins, min_durations)

    return session, metadata, transitions, activity


def append_transitions(concatenated_transitions:Dict[str,Dict], c_transitions:Dict)->Dict[str,Dict]:
    """
    Append to concatened_transitions activity and metadata for all transitions_name

    Parameters
    ----------
    concatenated_transitions : Dict[str,Dict]
        Dict with all transitions activity and metadata
    c_transitions : Dict
        current block of data to be append

    Returns
    -------
    Dict[str,Dict]
    """
    c_metadata = c_transitions['metadata']
    for transitions_name, c_activity in c_transitions['activity'].items():
        prev_activity = concatenated_transitions.get(transitions_name,{'activity':[],
                                                                       'metadata':[]})
        prev_activity['activity'].append(np.nanmean(c_activity, 2)) #Average for all same transitions for each neuron
        prev_activity['metadata'].append(c_metadata)
        concatenated_transitions[transitions_name] = prev_activity
    return concatenated_transitions


def merge_all_sessions(all_sessions:Dict[str,Dict])->Dict[str,Dict]:
    """
    During process_all_session, average each neuron for all transition of the same kind
    Merge all the sessions in the same dict

    Parameters
    ----------
    all_sessions : Dict[str,Dict]
        session_name :dict from :py:func:'process_session'

    Returns
    -------
    Dict[str,Dict]
        transition_name
            activity[n_neurons,nbins]
            metadata[n_neurons,Rat,Day,Shank,ID,Region,Type]
    """
    concatenated_transitions = {}
    for _, c_transitions in all_sessions.items():
        concatenated_transitions = append_transitions(concatenated_transitions, c_transitions)
    
    for transition_name,c_transitions in concatenated_transitions.items():
        concatenated_transitions[transition_name]['activity'] = np.vstack(c_transitions['activity'])
        concatenated_transitions[transition_name]['metadata'] = pd.concat(c_transitions['metadata'])


    return concatenated_transitions


def process_all_sessions(base_folder: Union[Path, str] = upath['base_folder'],
                         save=False,force = False, **kwargs) -> Tuple:
    """
    Run :py:func:'process_session' for all session in the dataset

    Parameters
    ----------
    base_folder : Union[Path,str], optional
        Path to the dataset, by default upath['base_folder']

    Returns
    -------
    _type_
        _description_
    # """

    session_list = load.session_list()
    all_sessions = {}
    for p in tqdm(session_list.Path):
        try:
            c_session, c_metadata, c_transitions, c_activity = process_session(local_path=p, 
                                                                               save=save, 
                                                                               force=force,
                                                                               **kwargs)
            all_sessions[c_session['session_name']] = {'metadata': c_metadata,
                                                       'transitions': c_transitions,
                                                       'activity': c_activity}
        except:
            print(f'{p} not taken care of because bug')

    merged = merge_all_sessions(all_sessions)
    io.save_shelve('processed_data/transitions',
                    {'merged_sessions':merged})

    return all_sessions

if __name__ == '__main__':

    save = True
    force = True
    
    all_session = process_all_sessions(min_durations = min_durations,nbins = states_nbins, save = save,force = force)
    # process_session(min_durations = min_durations,nbins = nbins, save = save,force = force)
    # cProfile.run('process_session(save= False,force = True,min_durations = min_durations,nbins=nbins)','run_transition')
    # p = pstats.Stats('run_transition')
    # p.sort_stats(SortKey.CUMULATIVE).print_stats(10)