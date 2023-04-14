from settings import upath

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from bk import load
from bk import stats
from bk import compute

import neuroseries as nts

from tqdm import tqdm
from functools import reduce

from pathlib import Path
from typing import Union, Optional,Tuple, Dict, Sequence
from numpy.typing import ArrayLike



def process_session(base_folder:Union[Path,str]= upath['base_folder'],local_path:Union[Path,str]=upath['example_session'],discarded_states:Sequence[str] = ('DROWSY','WAKE'))->pd.DataFrame:
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

    Returns
    -------
    pd.DataFrame
        Data frame concatenated from metadata, :py:func:`state_fr` and :py:func:`rem_on`
        
        Rat,Day,Shank,Id,Region,Type,SessID,NREM,REM,WAKE_HOMECAGE,tot_spikes,pInc,pDec,Surprise
        pInc is the probability of increased FR during REM
        pDec is the probability of decrease FR during REM
    """
    md = load.session(base_folder=base_folder,local_path=local_path)
    
    discarded_states = set(discarded_states)

    neurons,metadata = load.spikes(md)
    metadata['SessID'] = metadata.index

    id_columns = list(metadata.columns)

    states = load.sleep_scoring(md,discard = discarded_states)
    states['SLEEP'] = states['NREM'].union(states['REM'])


    df_states_fr = states_fr(neurons,metadata,states)
    df_rem_on = rem_on(neurons,metadata,states)
    df_deltas = delta_fr_extended_state(neurons,metadata,states,60,5,5)
    fr_extended = fr_across_extended(neurons,metadata,states,'BLA','Pyr',30)

    all_df = [df_states_fr,df_rem_on,df_deltas]
    df = reduce(lambda left,right: pd.merge(left,right,on=id_columns),all_df)

    return df,fr_extended

def process_all_sessions(base_folder:Union[Path,str]= upath['base_folder'])->Tuple:
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
            df,extended_fr = process_session(local_path=p)
            all_df.append(df)
            all_extended_fr.append(extended_fr)
        except:
            print(f'{p} not taken care of because bug')
    
    df = pd.concat(all_df)
    df.to_csv('processed_data/fr.csv',index = False)

    extended_fr = merge_extended(all_extended_fr)
    
    return df,extended_fr

def states_fr(neurons:ArrayLike,metadata:pd.DataFrame,states:Dict[str,nts.IntervalSet])->pd.DataFrame:
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
    for s,intervals in states.items():
        fr = [len(n.restrict(intervals))/intervals.tot_length(time_units = 's') 
              for n in neurons]
        fr_states[s] = fr
    
    tot_spikes = [len(n) for n in neurons]
    
    fr_states['tot_spikes'] = tot_spikes
    
    return pd.concat((metadata,fr_states),axis=1)

def rem_on(neurons:ArrayLike,metadata:pd.DataFrame,states:Dict[str,nts.IntervalSet])->pd.DataFrame:
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
    poisson = stats.poisson_test(neurons,states['REM'],states['NREM'])
    return pd.concat((metadata,poisson),axis = 1)

def fr_across_extended_one_state(neurons:ArrayLike,
                                 states:Dict[str,nts.IntervalSet],
                                 binSize:int,
                                 params:Dict[str,Dict],
                                 extended_states:nts.IntervalSet)->Dict[str,list]:
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
    across_sleep_fr = {}
    for state in params['sub_states']:
        across_sleep_fr[state] = []
        for s, e in extended_states.iloc:
            current_state = nts.IntervalSet(s, e)
            state_intervals = states[state].intersect(current_state)

            act = compute.binSpikes(neurons,binSize,as_Tsd = True).restrict(state_intervals)
            act = compute.nts_zscore(act)
            act.index = act.index - act.index.values[0]
            average_act = nts.Tsd(act.times(),np.nanmean(act.values,1))

            across_sleep_fr[state].append(average_act)

    return across_sleep_fr

def fr_across_extended(neurons:ArrayLike,
                       metadata:pd.DataFrame,
                       states:Dict[str,nts.IntervalSet],
                       stru:str,
                       types:str,
                       binSize:int)->Dict[str,list]:
    """
    Compute zscore FR for all extended state of a session

    Parameters
    ----------
    neurons : ArrayLike
        list of neurons given by :py:func:`load.spikes`
    metadata : pd.DataFrame
        metadata as :py:func:`load.spikes`
    states : Dict[str,nts.IntervalSet]
        states as given by :py:func:`load.intervals`
    stru : str
        Structure of interest (BLA, HPC, etc )
    types : str
        Type of neurons (Pyr, Int)
    binSize : int
        Size of the bin used to compute the FR


    Returns
    -------
    Dict[str,list]
    """
    extended_params = {'sleep':{'sleep_th':60*30,
                            'wake_th':60,
                            'sub_states':['NREM','REM']},
                        'wake':{'sleep_th':60,
                            'wake_th':60*30,
                            'sub_states':['WAKE_HOMECAGE']}}
    
    neurons = neurons[(metadata.Region == stru) & (metadata.Type == types)]

    across_sleep_fr = {}
    for state_name,params in extended_params.items():
        extended_states = compute.extended(states,state_name,params['sleep_th'],params['wake_th'])
        across_sleep_fr[state_name] = fr_across_extended_one_state(neurons, states, binSize, params, extended_states)

    return across_sleep_fr

def delta_fr_one_state(neurons:ArrayLike, 
                       states:Dict[str,nts.IntervalSet], 
                       length_compute_fr:int, 
                       params:Dict[str,Dict], 
                       extended_states:nts.IntervalSet)->Dict[str,list]:
    """
    Compute the Delta FR between the beg and the end of a state extended
    Parameters
    ----------
    neurons : ArrayLike
        list of neurons given by :py:func:`load.spikes`
    states : Dict[str,nts.IntervalSet]
        states as given by :py:func:`load.intervals`
    length_compute_fr : int
        Amount of time taken at the beg and end to compute the FR
    params : Dict[Dict]
        Params given by :py:func:`delta_fr_extended_state`
    extended_states : nts.IntervalSet
        extended_states as given by :py:func:`compute.extended`

    Returns
    -------
    Dict
        Dict of list length N neurons, contains zscore delta FR for each substate
    """
    deltas = {}
    for state in params['sub_states']:
        deltas[state] = []
        for s, e in extended_states.iloc:
            current_state = nts.IntervalSet(s, e)
            state_intervals = states[state].intersect(current_state)
            act = compute.binSpikes(neurons,1,as_Tsd = True).restrict(state_intervals)
            act = compute.nts_zscore(act)
            fr_beg = np.nansum(act.iloc[0:length_compute_fr].values,0)/length_compute_fr
            fr_end = np.nansum(act.iloc[-length_compute_fr:-1].values,0)/length_compute_fr
            deltas[state].append(fr_end-fr_beg)
    
    return deltas

def delta_fr_extended_state(neurons:ArrayLike,
                            metadata:pd.DataFrame,
                            states:Dict[str,nts.IntervalSet],
                            length_compute_fr:int,
                            sleep_th:int,
                            wake_th:int)->pd.DataFrame:
    """
    Compute the Delta FR between the beg and the end of a list of extended state

    Parameters
    ----------
    neurons : ArrayLike
        list of neurons given by :py:func:`load.spikes`
    metadata : pd.DataFrame
        metadata as :py:func:`load.spikes`
    states : Dict[str,nts.IntervalSet]
        states as given by :py:func:`load.intervals`
    length_compute_fr : int
        Amount of time taken at the beg and end to compute the FR
    sleep_th : int
        _description_
    wake_th : int
        _description_

    Returns
    -------
    pd.DataFrame
        DataFrame with columns NREM/REM/WAKE_HOMECAGE, each line is a neuron and values are delta FR between the beg and end of extended
    """
    extended_params = {'sleep':{'sleep_th':60*30,
                                'wake_th':60,
                                'sub_states':['NREM','REM']},
                       'wake':{'sleep_th':60,
                                'wake_th':60*30,
                                'sub_states':['WAKE_HOMECAGE']}}

    
    deltas = {}
    for state_name,params in extended_params.items():
        extended_states = compute.extended(states,state_name,params['sleep_th'],params['wake_th'])
        deltas[state_name] = delta_fr_one_state(neurons, states, length_compute_fr, params, extended_states)

    df = pd.DataFrame()
    for state in deltas:
        for substate in deltas[state]:
            df[f'delta_{substate}'] = np.nanmean(deltas[state][substate],0)
    return pd.concat((metadata,df),axis = 1)

def merge_extended(all_extended_fr:list[Dict])->Dict:
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
    activiy = {'NREM': [],
                'REM': [],
                'WAKE_HOMECAGE': []}
    for sess in all_extended_fr:
        for state in sess:
            for substate in sess[state]:
                for v in sess[state][substate]:
                    activiy[substate].append(v)
    return activiy

if __name__ == "__main__":
    df,fr_across_extended = process_all_sessions()
    # FIXME : SAVE DATA
