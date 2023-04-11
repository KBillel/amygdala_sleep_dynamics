from bk import load
from bk import stats

import neuroseries as nts

from pathlib import Path
from typing import Union, Optional,Tuple, Dict, Sequence
from numpy.typing import ArrayLike

import pandas as pd

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
    wake_useless = False
    if 'WAKE' in discarded_states:
        wake_useless = True
        discarded_states.discard('WAKE')

    neurons,metadata = load.spikes(md)
    metadata['SessID'] = metadata.index

    id_columns = list(metadata.columns)

    states = load.intervals(md,'Intervals/sleep_scoring.csv',discarded_states)
    homecage = load.homecage(md)
    states['WAKE_HOMECAGE'] = homecage.intersect(states['WAKE'])
    states['SLEEP'] = states['NREM'].union(states['REM'])
    
    if wake_useless:
        states.pop('WAKE')

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


if __name__ == "__main__":
    df = process_session('/mnt/electrophy/Gabrielle/GG-Dataset-Light/','Rat08/Rat08-20130713',('DROWSY','WAKE'))
    # FIXME : SAVE DATA
