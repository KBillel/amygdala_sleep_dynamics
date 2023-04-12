from bk import load
from bk import stats
from bk import compute

import neuroseries as nts

from pathlib import Path
from typing import Union, Optional,Tuple, Dict, Sequence
from numpy.typing import ArrayLike

import pandas as pd
import numpy as np

def process_session(base_folder:Union[Path,str],local_path:Union[Path,str],discarded_states:Sequence[str] = ('DROWSY','WAKE'))->pd.DataFrame:
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

    df = pd.merge(df_states_fr,df_rem_on,on=id_columns)
    
    return df



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

def delta_fr_extended_state(neurons:ArrayLike,
                            metadata:pd.DataFrame,
                            states:Dict[str,nts.IntervalSet],
                            length_compute_fr:int,
                            sleep_th:int,
                            wake_th:int)->pd.DataFrame:
    
    extended_params = {'sleep':{'sleep_th':60*30,
                                'wake_th':30,
                                'sub_states':['NREM','REM']},
                       'wake':{'sleep_th':30,
                                'wake_th':60*30,
                                'sub_states':['WAKE_HOMECAGE']}}

    
    deltas = {}
    for state_name,params in extended_params.items():
        extended_states = compute.extended(states,state_name,params['sleep_th'],params['wake_th'])
        deltas[state_name] = delta_fr_one_state(neurons, states, length_compute_fr, params, extended_states)

    return deltas

def delta_fr_one_state(neurons, states, length_compute_fr, params, extended_states):
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


if __name__ == "__main__":
    df = process_session('/mnt/electrophy/Gabrielle/GG-Dataset-Light/','Rat08/Rat08-20130713',('DROWSY','WAKE'))
    
    md = load.session()
    neurons, metadata = load.spikes(md)
    states = load.sleep_scoring(md)

    

    # FIXME : SAVE DATA
