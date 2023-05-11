from settings import upath,min_durations

import json
from hashlib import sha256
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats

from bk import load
from bk import stats
from bk import compute
from bk import io

import neuroseries as nts

from tqdm import tqdm
from functools import reduce, wraps
import inspect
import shelve

from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Sequence, List
from numpy.typing import ArrayLike

FORCE = False


def check_json(json_path, saved_args):
    with open(json_path, 'r') as jf:
        old_args = json.load(jf)

    for name, value in saved_args.items():
        old_value = old_args.get(name)
        if old_value != value:
            return False
    return True


def df_saver(args_to_save=None, force=False):
    if args_to_save is None:
        args_to_save = []

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            id_columns = ['Rat', 'Day', 'Shank', 'Id']

            source = inspect.getsource(func)
            hash = sha256(source.encode()).hexdigest()

            csv_path = Path(f'processed_data/{func.__name__}.csv')
            json_path = csv_path.with_suffix('.json')

            params = list(inspect.signature(func).parameters.keys())
            nargs = len(args)

            all_args = {p: v for p, v in zip(params, args[0:nargs])}
            all_args.update(kwargs)

            saved_args = {k: v for k, v in all_args.items()
                          if k in args_to_save}
            saved_args['saved_path'] = csv_path.as_posix()
            saved_args['sha256'] = hash
            valid = False
            tmp_csv = None
            if csv_path.exists():
                if json_path.exists():
                    valid = check_json(json_path, saved_args)
                if valid:
                    tmp_csv = pd.read_csv(csv_path)
                    metadata = all_args.get('metadata', pd.DataFrame())
                    day_num = metadata['Day'].unique()[0]
                    rat_num = metadata['Rat'].unique()[0]
                    mask_session = (tmp_csv['Rat'] == rat_num) & (
                        tmp_csv['Day'] == day_num)
                    f_session = tmp_csv[mask_session]
                    # if session is already processed and we do not need to force -> Return whats in the dataframe
                    if len(f_session) > 0 and not force:
                        return f_session
                    elif len(f_session) > 0 and force:
                        tmp_csv = tmp_csv[np.logical_not(mask_session)]

            df = func(*args, **kwargs)
            if tmp_csv is not None:
                df = pd.concat([tmp_csv, df])
            df.to_csv(csv_path, index=False)

            with open(f'processed_data/{func.__name__}.json', 'w') as jf:
                json.dump(saved_args, jf, indent=2)

            return df

        return wrapper
    return decorator


def rebin_fr(act: nts.TsdFrame, binSize: int = 30):
    binSize *= 1e6
    bins = np.arange(np.min(act.times()), np.max(act.times()), binSize)

    val, bin_edge, _ = scipy.stats.binned_statistic(act.times(), act.values.T, 'mean', bins)

    t = (bin_edge[:-1] + bin_edge[1:]) / 2
    binned_act = nts.TsdFrame(t, val.T)

    return binned_act


def rebin_extended(fr_extended, binSize=30):
    binned_act = {}
    for state, sub_states in fr_extended.items():
        binned_act[state] = {}
        for c_sub, list_act in sub_states.items():
            binned_act[state][c_sub] = []
            for act in list_act:
                c_binned_act = rebin_fr(act, 30)
                binned_act[state][c_sub].append(c_binned_act)
    return binned_act


def fr_across_extended_one_state(neurons: ArrayLike,
                                 states: Dict[str, nts.IntervalSet],
                                 binSize: int,
                                 sub_states: List[str],
                                 extended_states: nts.IntervalSet) -> Dict[str, list]:
    """
    Compute the zscore FR during a list of extended state


    Parameters
    ----------
    neurons : ArrayLike
        list of neurons given by :py:func:`load.spikes`
    states : Dict[str,nts.IntervalSet]
        states as given by :py:func:`load.intervals`
    binSize : int
        Size of the bin used to compute the FR
    params : Dict[str,Dict]
        Params given by :py:func:`fr_across_extended`

    extended_states : nts.IntervalSet
        extended_states as given by :py:func:`compute.extended`

    Returns
    -------
    Dict[str,list]
        _description_
    """

    across_extended_fr = {}
    for state in sub_states:
        across_extended_fr[state] = []
        for s, e in extended_states.iloc:
            current_state = nts.IntervalSet(s, e)
            state_intervals = states[state].intersect(current_state)
            # Only REM/NREM or WAKE_HOMECAGE of this extended
            act = compute.binSpikes(neurons, binSize, as_Tsd=True).restrict(state_intervals)
            act = compute.nts_zscore(act)
            # act.index = act.times() - act.index.values[0] # Is this wrong #TODO check if wrong IMPORTANT
            act.index = act.times() - s
            # Reset time of act to zero to start counting from beggining of extended
            # average_act = nts.Tsd(act.times(),np.nanmean(act.values,1))

            across_extended_fr[state].append(act)

    return across_extended_fr


def fr_across_extended(neurons: ArrayLike,
                       metadata: pd.DataFrame,
                       extended_params: Dict[str, Dict],
                       states: Dict[str, nts.IntervalSet],
                       binSize: int) -> Dict[str, list]:
    """
    Compute zscore FR for all extended state of a session

    Parameters
    ----------
    neurons : ArrayLike
        list of neurons given by :py:func:`load.spikes`
    metadata : pd.DataFrame
        metadata as :py:func:`load.spikes`
    extended_states : Dict[str,nts.IntervalState]
        dict of all extended_sleep/wake for the session
    states : Dict[str,nts.IntervalSet]
        states as given by :py:func:`load.intervals`
    binSize : int
        Size of the bin used to compute the FR
    Returns
    -------
    Dict[str,list]
    """

    across_sleep_fr = {}
    extended_states = {}
    for state_name, params in extended_params.items():
        current_extended = compute.extended(states, state_name, params['sleep_th'], params['wake_th'])
        across_sleep_fr[state_name] = fr_across_extended_one_state(neurons, states, binSize, params['sub_states'], current_extended)
        extended_states[state_name] = current_extended
    return across_sleep_fr


@df_saver(force=FORCE)
def states_fr(neurons: ArrayLike, metadata: pd.DataFrame, states: Dict[str, nts.IntervalSet]) -> pd.DataFrame:
    """
    Return the Firing rates in all states

    Parameters
    ----------
    neurons : ArrayLike
        list of neurons given by :py:func:`load.spikes`
    metadata : pd.DataFrame
        metadata as :py:func:`load.spikes`
    states : Dict[str,nts.Intervalset]
        states as given by :py:func:`load.intervals`

    Returns
    -------
    pd.DataFrame
        dataframes with the firing rates
    """
    fr_states = pd.DataFrame()
    for s, intervals in states.items():
        fr = [len(n.restrict(intervals))/intervals.tot_length(time_units='s')
              for n in neurons]
        fr_states[s] = fr

    tot_spikes = [len(n) for n in neurons]

    fr_states['tot_spikes'] = tot_spikes

    return pd.concat((metadata, fr_states), axis=1)

@df_saver(force=FORCE)
def rem_on(neurons: ArrayLike, metadata: pd.DataFrame, states: Dict[str, nts.IntervalSet]) -> pd.DataFrame:
    """
    Function that compute if a neuron is firing statisticaly more during REM sleep using the poisson test

    Parameters
    ----------
    neurons : ArrayLike
        list of neurons given by :py:func:`load.spikes`
    metadata : pd.DataFrame
        metadata as :py:func:`load.spikes`
    states : Dict[str,nts.Intervalset]
        states as given by :py:func:`load.intervals`

    Returns
    -------
    pd.DataFrame
        pInc : probability of increased FR
        pDec : probability of decreased FR
        surprise : log ratio of pInc / pDec
    """
    poisson = stats.poisson_test(neurons, states['REM'], states['NREM'])
    return pd.concat((metadata, poisson), axis=1)

@df_saver(args_to_save=['length_to_compute'], force=FORCE)
def delta_extended(fr_extended, metadata, length_to_compute):
    length_to_compute *= 1e6
    dfs = []
    for state, sub_states in fr_extended.items():
        for c_sub, list_act in sub_states.items():
            l_deltas = []
            for act in list_act:
                binSize = int(np.diff(act.times())[0])
                nbins = int(length_to_compute/binSize)
                c_delta = np.nanmean(
                    act.values[-nbins-1:-1], 0) - np.nanmean(act.values[0:nbins, :], 0)
                l_deltas.append(c_delta)
            if len(l_deltas) != 0:
                dfs.append(pd.DataFrame(np.nanmean(
                    l_deltas, axis=0), columns=[c_sub]))
            else:
                dfs.append(pd.DataFrame(columns=[c_sub]))
    df = pd.concat(dfs, axis=1)
    df = pd.concat([metadata, df], axis=1)
    return df

# @df_saver(args_to_save=['length_to_compute'],force = FORCE)
def effect_extended(neurons: ArrayLike, metadata: pd.DataFrame,states: Dict[str, nts.IntervalSet],params,length_to_compute):
    length_to_compute *= 1e6
    extended_sleep = compute.extended(states, 'sleep', params['sleep_th'], params['wake_th'])
    for i,c_extended in extended_sleep.iterrows():
        before = nts.IntervalSet(c_extended.start-length_to_compute,c_extended.start).intersect(states['WAKE_HOMECAGE'])
        after = nts.IntervalSet(c_extended.end,c_extended+length_to_compute).intersect(states['WAKE_HOMECAGE'])
        print(before,after)


def collapse_substates(extended:Dict[str,list])->Dict[str,list]:
    """
    _summary_

    Parameters
    ----------
    extended : Dict[str,list]
        _description_

    Returns
    -------
    Dict[str,list]
        _description_
    """
    all_extended = {}
    for _,all_substates in extended.items():
        all_extended.update({substate:fr for substate,fr in all_substates.items()})
    return all_extended

def concat_substates(extended:Dict[str,list],c_extended,c_metadata:pd.DataFrame)->Dict[str,Dict]:
    for substate,l_fr in c_extended.items():
        prev_extended = extended.get(substate,{'FR':[],
                                               'metadata':[]})
        prev_extended['FR'].extend(l_fr)
        prev_extended['metadata'].extend([c_metadata]*len(l_fr))
        extended[substate] = prev_extended
    return extended

def merge_extended(all_extended_fr: Dict[str,Dict],all_metadata:Dict[str,Dict]) -> Dict:
    """
    Merge output of :py:func:`process_all_session` into a simple Dict

    Parameters
    ----------
    all_extended_fr : list[Dict]
        _description_

    Returns
    -------
    Dict
        NREM | REM | WAKE_HOMECAGE
        list of firing for extended sleep / wake
    """
    extended = {}
    for session_name,c_extended in all_extended_fr.items():
        c_metadata = all_metadata[session_name]
        collapsed_extended = collapse_substates(c_extended)
        extended = concat_substates(extended,collapsed_extended,c_metadata)
    
    return extended

def save_data(session, metadata, binned_fr_extended,params):
    """
    Save data of :py:func:'process_session' in a shelve

    Parameters
    ----------
    session : _type_
    metadata : _type_
    binned_fr_extended : _type_
    params : _type_
    """

    d = {'unique_sessions':{session['session_name'] :{'session':session,
                                                      'FR':binned_fr_extended,
                                                      'metadata':metadata,
                                                      'params':params}}}
    
    io.save_shelve('processed_data/binned_fr_extended',dict = d,params = params)

def process_session(base_folder: Union[Path, str] = upath['base_folder'],
                    local_path: Union[Path, str] = upath['example_session'],
                    discarded_states: Sequence[str] = ('DROWSY', 'WAKE'),
                    binSize: int = 1,
                    params:Dict = {},
                    save:bool = False) -> Tuple[Dict,pd.DataFrame,Dict[str,Dict],pd.DataFrame]:
    """
    Process a session with computation relative to states firing rates

    Parameters
    ----------
    base_folder : Union[Path,str]
        Path to the dataset
    local_path : Union[Path,str]
        relative path to the session
    discarded_states : Sequence[str], optional
        states not to be computed, by default ('DROWSY','WAKE')
    binSize : int
        binSize used for computing the firing rates across extended_sleep/wake

    Returns
    -------
    pd.DataFrame
        Data frame concatenated from metadata, :py:func:`state_fr` and :py:func:`rem_on`

        Rat,Day,Shank,Id,Region,Type,SessID,NREM,REM,WAKE_HOMECAGE,tot_spikes,pInc,pDec,Surprise
        pInc is the probability of increased FR during REM
        pDec is the probability of decrease FR during REM
    """

    session = load.session(base_folder=base_folder, local_path=local_path)
    discarded_states = set(discarded_states)

    neurons, metadata = load.spikes(session)
    metadata['SessID'] = metadata.index

    id_columns = list(metadata.columns)

    states = load.sleep_scoring(session, discard=discarded_states,drop_short_intervals=min_durations)
    states['SLEEP'] = states['NREM'].union(states['REM'])

    fr_extended = fr_across_extended(neurons, metadata, params, states, binSize)
    binned_fr_extended = rebin_extended(fr_extended, 30)
    

    df_states_fr = states_fr(neurons, metadata, states)
    df_rem_on = rem_on(neurons, metadata, states)
    df_deltas = delta_extended(fr_extended, metadata, 30)
    df_effect_extended_sleep = effect_extended(neurons,metadata,states,params['sleep'],30)

    all_df = [df_states_fr, df_rem_on, df_deltas]
    df = reduce(lambda left, right: pd.merge(
        left, right, on=id_columns), all_df)

    if save:
        save_data(session, metadata, binned_fr_extended,params)

    return session,df,binned_fr_extended,metadata

def process_all_sessions(base_folder: Union[Path, str] = upath['base_folder'],params = {},save = False,**kwargs) -> Tuple:
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
    l_all_df = []
    all_extended_fr = {}
    all_metadata = {}
    
    for p in tqdm(session_list.Path):
        try:
            # -> Stuff are saved using the decorator here
            session,df, extended_fr,metadata = process_session(local_path=p,params = params,save=save)
            
            l_all_df.append(df)
            all_extended_fr[session['session_name']] = extended_fr
            all_metadata[session['session_name']] = metadata
        except:
            print(f'{p} not taken care of because bug')
    

    all_df = pd.concat(l_all_df)
    merged_extended = merge_extended(all_extended_fr,all_metadata)
    io.save_shelve('processed_data/binned_fr_extended',
                   {'merged_sessions':merged_extended})

    return all_df, merged_extended

if __name__ == "__main__":

    params = {'sleep': {'sleep_th': 60*30,
                        'wake_th': 60,
                        'sub_states': ['NREM', 'REM']},
              'wake': {'sleep_th': 60,
                       'wake_th': 60*30,
                       'sub_states': ['WAKE_HOMECAGE']}}

    session = load.session()
    discarded_states = set(['DROWSY','WAKE'])

    neurons, metadata = load.spikes(session)
    metadata['SessID'] = metadata.index

    id_columns = list(metadata.columns)

    states = load.sleep_scoring(session, discard=discarded_states,drop_short_intervals=min_durations)
    states['SLEEP'] = states['NREM'].union(states['REM'])

    df_effect_extended_sleep = effect_extended(neurons,metadata,states,params['sleep'],5*60)




    # save = True
    # process_all_sessions(params = params, save = True)
    # process_session(params=params, save = True)
