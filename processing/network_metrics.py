from pathlib import Path
from typing import Union, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import zscore
from tqdm import tqdm

import neuroseries as nts
from bk import compute
from bk import io
from bk import load
from bk import misc
from processing.transitions import find_transitions
from settings import upath, min_durations, states_nbins, network_metrics_params


def check_session(metadata,stru,min_pyr,min_int):
    npyr =  np.sum((metadata.Region == stru) & (metadata.Type == 'Pyr'))
    nint = np.sum((metadata.Region == stru) & (metadata.Type == 'Int'))

    if (npyr < min_pyr) or (nint<min_int):
        return False
    else:
        return True


def compute_fr(neurons:ArrayLike,
                metadata:pd.DataFrame,
                stru:str,
                binSize:float,
                start:Optional[float] = 0,
                stop:Optional[float] = None,
                nbins:Optional[int] = None)->ArrayLike:
    """
    Compute firing rates for Pyr and Interneurons

    Parameters
    ----------
    neurons : ArrayLike
        list of neurons given by :py:func:`load.spikes`
    metadata : pd.DataFrame
        metadata as :py:func:`load.spikes`
    stru : str
        structure (BLA/Hpc/Pir) to select neurons from
    binSize : float
        binSize used for binning the spikes

    Returns
    -------
    ArrayLike
        two 1d vector
    """
    if not check_session(metadata,stru,10,3):
        return np.array([np.nan]),np.array([np.nan])

    t,bin_matrix = compute.binSpikes(neurons,binSize,start=start,stop=stop,nbins=nbins)
    f_bin_matrix_pyr,_ = misc.filter_neurons(bin_matrix,metadata,stru,'Pyr',True)
    f_bin_matrix_int,_ = misc.filter_neurons(bin_matrix,metadata,stru,'Int',True)
    
    m_pyr = np.nanmean(f_bin_matrix_pyr,0)
    m_int = np.nanmean(f_bin_matrix_int,0)

    return t,m_pyr,m_int



def compute_eib(neurons:ArrayLike,
                metadata:pd.DataFrame,
                stru:str,
                binSize:float,
                start:Optional[float] = 0,
                stop:Optional[float] = None,
                nbins:Optional[int] = None)->ArrayLike:
    """
    Compute excitatory inhibitory balance across time for all neurons in stru.

    Parameters
    ----------
    neurons : ArrayLike
        list of neurons given by :py:func:`load.spikes`
    metadata : pd.DataFrame
        metadata as :py:func:`load.spikes`
    stru : str
        structure (BLA/Hpc/Pir) to select neurons from
    binSize : float
        binSize used for binning the spikes

    Returns
    -------
    eib: ArrayLike
        1d vector with EIB for the whole session
    """
    if not check_session(metadata,stru,10,3):
        return np.array([np.nan]),np.array([np.nan])

    t,bin_matrix = compute.binSpikes(neurons,binSize,start=start,stop=stop,nbins=nbins)
    f_bin_matrix_pyr,_ = misc.filter_neurons(bin_matrix,metadata,stru,'Pyr',True)
    f_bin_matrix_int,_ = misc.filter_neurons(bin_matrix,metadata,stru,'Int',True)
    
    m_pyr = np.nanmean(f_bin_matrix_pyr,0)
    m_int = np.nanmean(f_bin_matrix_int,0)

    eib = m_pyr / (m_pyr+m_int)
    return t,eib

def compute_cv(neurons:ArrayLike,
                metadata:pd.DataFrame,
                stru:str,binSize:float,
                start:Optional[float] = 0,
                stop:Optional[float] = None,
                nbins:Optional[int] = None)->ArrayLike:
    """
    Compute coefficient of variations across principal neurons across time for all neurons in stru.

    Parameters
    ----------
    neurons : ArrayLike
        list of neurons given by :py:func:`load.spikes`
    metadata : pd.DataFrame
        metadata as :py:func:`load.spikes`
    stru : str
        structure (BLA/Hpc/Pir) to select neurons from
    binSize : float
        binSize used for binning the spikes

    Returns
    -------
    CV: ArrayLike
        1d vector with CV for the whole session
    """
    if not check_session(metadata,stru,10,0):
        return np.array([np.nan]),np.array([np.nan])
    t,bin_matrix = compute.binSpikes(neurons,binSize,start=start,stop=stop,nbins=nbins)
    f_bin_matrix_pyr,_ = misc.filter_neurons(bin_matrix,metadata,stru,'Pyr',True)
    cv = np.std(f_bin_matrix_pyr,0) / np.nanmean(f_bin_matrix_pyr,0)
    return t,cv

def compute_sync(bin_matrix:ArrayLike)->float:
    """
    Compute a single value of sync from a neurons x bins matrix

    Parameters
    ----------
    bin_matrix : ArrayLike
        matrix as given by :py:func:'compute.binSpikes'

    Returns
    -------
    float
        value of sync for this matrix
    """
    corr = np.corrcoef(bin_matrix)
    np.fill_diagonal(corr,np.nan)
    sync = np.nanmean(corr)
    return sync

def compute_sync_moving_windows(neurons:ArrayLike,
                                metadata:pd.DataFrame,
                                stru:str,
                                binSize:float,
                                winSize:float,
                                step:float,
                                start:Optional[float] = 0,
                                stop:Optional[float] = None,
                                nbins:Optional[int] = None)->ArrayLike:
    """
    Compute synchrony of principal neurons in stru across time. 

    Parameters
    ----------
    neurons : ArrayLike
        list of neurons given by :py:func:`load.spikes`
    metadata : pd.DataFrame
        metadata as :py:func:`load.spikes`
    stru : str
        structure (BLA/Hpc/Pir) to select neurons from
    binSize : float
        binSize used for binning the spikes
    winSize : float
        Size of the window to compute sync in seconds
    step : float
        step of the sliding window. If step = winSize there is no overlap

    Returns
    -------
    ArrayLike
        1d vector with sync for the whole session
    """
    if not check_session(metadata,stru,10,0):
        return np.array([np.nan]),np.array([np.nan])
    
    t,bin_matrix = compute.binSpikes(neurons,binSize,start=start,stop=stop)

    # winSize = int(winSize/binSize) #Convert second to idx
    # step = int(step / binSize)
    binSize = np.median(np.diff(t))
    
    winSize = int(winSize/binSize)
    step = int(step / binSize)
    if nbins is not None:
        winSize = int(len(t)/nbins)
        step = winSize

    f_bin_matrix_pyr,_ = misc.filter_neurons(bin_matrix,metadata,stru,'Pyr',True)

    window_strides = np.lib.stride_tricks.sliding_window_view(f_bin_matrix_pyr,winSize,axis = 1)
    window_strides = window_strides[:,::step,:].transpose((1,0,2))

    t_window_strides = np.lib.stride_tricks.sliding_window_view(t,winSize)
    t_window_strides = t_window_strides[::step,:]

    sync = np.array([compute_sync(win) for win in window_strides])
    t_sync = np.nanmean(t_window_strides,1)

    return t_sync,sync


def make_epochs_average(metrics:Dict[str,Tuple[ArrayLike,ArrayLike]],states:Dict[str,nts.IntervalSet],nbins:Dict[str,int]):
    """
    Take raw value of metrics and average them in nbins by states.

    Parameters
    ----------
    metrics : Dict[str,Tuple[ArrayLike,ArrayLike]]
        Dict with eib,zeib,sync and cv raw value for the whole session. 
    states : Dict[str,nts.IntervalSet]
        states as given by :py:func:`bk.load.sleep_scoring`
    nbins : Dict[str,int]
        either int or Dict with state:nbins

    Returns
    -------
    pd.DataFrame
        
    """
    if isinstance(nbins,int):
        nbins = {state:nbins for state in states.keys()}
    long_states = pd.DataFrame(pd.concat(states))
    make_bins_average(metrics, long_states,nbins)

    return long_states

def make_bins_average(metrics:Dict[str,Tuple[ArrayLike,ArrayLike]],states:Dict[str,nts.IntervalSet],nbins:Dict[str,int]):
    """
    Fonction that actually does the job of averaging

    Parameters
    ----------
    metrics : Dict[str,Tuple[ArrayLike,ArrayLike]]
        _description_
    states : Dict[str,nts.IntervalSet]
        _description_
    nbins : Dict[str,int]
        _description_
    """
    for metric,(t,value) in metrics.items():
        t_us = t * 1_000_000 # converting to µs
        l_metric = []
        for irow,row in states.iterrows():
            bins = np.linspace(row.start,row.end,nbins.get(row.state,1)+1)
            idx = np.searchsorted(t_us,bins)
            l_values = np.array([np.nanmean(value[i:j]) for i,j in zip(idx[:-1],idx[1:])])
            l_metric.append(l_values.squeeze())

        states[metric] = l_metric


def compute_metrics_at_transitions(neurons,
                                   metadata,
                                   stru,
                                   transitions,
                                   nbins,
                                   func,**kwargs):
    metric = {}
    for tr_name, tr_l_intervals in transitions.items():
        metric_array_tr = []
        for interval in tr_l_intervals:
            l_metric = []
            for irow, row in interval.iterrows():
                start_s = (row.start / 1_000_000)
                end_s = (row.end / 1_000_000)
                t,c_metric = func(neurons = neurons,
                                metadata = metadata,
                                stru = stru,
                                start = start_s,
                                stop = end_s,
                                nbins=nbins[row['state']],
                                **kwargs)
                l_metric.append(c_metric)

            metric_array_intervals = np.hstack(l_metric)
            metric_array_tr.append(metric_array_intervals)
        metric[tr_name] = np.vstack(metric_array_tr)
    return metric
    
        
    

def process_session(base_folder: Union[Path, str] = upath['base_folder'],
                    local_path: Union[Path, str] = upath['example_session'],
                    stru:str = 'BLA',
                    min_durations: Dict[str, int] = None,
                    save: bool = False,
                    force: bool = False,
                    nbins = states_nbins,
                    params = network_metrics_params) -> pd.DataFrame:
    """
    Process a session with computation relative to network metrics


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
    """

    session = load.session(base_folder, local_path)
    
    if not force:
        data = io.load_shelve('processed_data/network_metrics')
        if ('unique_sessions' in data) and (session['session_name'] in data['unique_sessions']):
            c_data = data['unique_sessions'][session['session_name']]
            return c_data['session'],c_data['metrics']
    
    
    states = load.sleep_scoring(session,drop_short_intervals = min_durations)
    neurons, metadata = load.spikes(session)

    metrics = {'raw':{},
               'examples':{},
               'averaged':{},
               'at_transitions':{}}
    
    metrics['examples']['FR'] = compute_fr(neurons,metadata,stru,binSize = params['eib']['binSize'])
    metrics['examples']['States'] = states
    metrics['raw']['eib'] = compute_eib(neurons,metadata,stru,binSize = params['eib']['binSize'])
    metrics['raw']['z_eib'] = metrics['raw']['eib'][0],zscore(metrics['raw']['eib'][1])
    metrics['raw']['cv'] = compute_cv(neurons,metadata,stru,binSize = params['cv']['binSize'])
    metrics['raw']['sync'] = compute_sync_moving_windows(neurons = neurons,
                                                         metadata = metadata,
                                                         stru = stru,
                                                         binSize = params['sync']['binSize'],
                                                         winSize = params['sync']['winSize'],
                                                         step = params['sync']['step'])
    
    metrics['averaged']['epochs'] = make_epochs_average(metrics['raw'],states,1)
    metrics['averaged']['thirds'] = make_epochs_average(metrics['raw'],states,3)
    metrics['averaged']['nbins'] = make_epochs_average(metrics['raw'],states,nbins)


    transitions = find_transitions(states,1,min_durations = min_durations)
    transitions.update(find_transitions(states, n_states=2, min_durations=min_durations))
    transitions.update(find_transitions(states, n_states=3, min_durations=min_durations))
    metrics['at_transitions']['eib'] = compute_metrics_at_transitions(neurons,
                                                                      metadata,
                                                                      stru,
                                                                      transitions,
                                                                      nbins,
                                                                      func = compute_eib,
                                                                      binSize = None)
    
    metrics['at_transitions']['cv'] = compute_metrics_at_transitions(neurons,
                                                                    metadata,
                                                                    stru,
                                                                    transitions,
                                                                    nbins,
                                                                    func = compute_cv,
                                                                    binSize = None)

    metrics['at_transitions']['sync'] = compute_metrics_at_transitions(neurons,
                                                                    metadata,
                                                                    stru,
                                                                    transitions,
                                                                    nbins,
                                                                    func = compute_sync_moving_windows,
                                                                    binSize = params['sync']['binSize'],
                                                                    winSize = params['sync']['winSize'],
                                                                    step = params['sync']['winSize'])


    if save:
        save_data(session,metrics,params)

    return session,metrics

def append_averaged_session(concatenated_session,session):
    metric_names = session['raw'].keys()
    
    for average_kind,df in session['averaged'].items():
        prev_concat_session = concatenated_session.get(average_kind,{m_name:{'NREM':[],
                                                                             'REM':[],
                                                                             'WAKE_HOMECAGE':[]} for m_name in metric_names})
        for m_name in metric_names:
            for state in prev_concat_session[m_name].keys():
                prev_concat_session[m_name][state].extend(df[m_name][df['state'] == state].values)
        concatenated_session[average_kind] = prev_concat_session
    return concatenated_session

def clean_metrics(metrics):
    for metric_name,all_state in metrics.items():
        for state_name,values in all_state.items():
            stack = np.atleast_2d(np.vstack(values))
            stack = stack[~np.all(np.isnan(stack),1),:].squeeze()
            metrics[metric_name][state_name] = stack

    return metrics

def append_transitions(concatenated_transitions,c_session):
    for metric_name, all_transitions in c_session['at_transitions'].items():
        prev_metric = concatenated_transitions.get(metric_name,{})
        for transitions_name, c_activity in all_transitions.items():
            prev_trans = prev_metric.get(transitions_name,[])
            if not np.all(np.isnan(c_activity)):
                prev_trans.append(c_activity)
            prev_metric[transitions_name] = prev_trans
        concatenated_transitions[metric_name] = prev_metric
    return concatenated_transitions

def merge_all_sessions(all_sessions:Dict[str,Dict])->Dict[str,Dict]:
    """
    During process_all_session, average all session into a single dict with dataframe or nparray
    Merge all the sessions in the same dict

    Parameters
    ----------
    all_sessions : Dict[str,Dict]
        session_name :dict from :py:func:'process_session'

    Returns
    -------
    Dict[str,Dict]
        ???
    """
    concatenated_sessions = {}
    concatenated_transitions = {}
    for _, c_session in all_sessions.items():
        concatenated_sessions = append_averaged_session(concatenated_sessions,c_session)
        concatenated_transitions = append_transitions(concatenated_transitions, c_session)

    
    for average_kind,all_metrics in concatenated_sessions.items():
        all_metrics = clean_metrics(all_metrics)
    
    for metric_name,all_transitions in concatenated_transitions.items():
        to_drop = []
        for transition_name,c_transitions in all_transitions.items():
            if len(c_transitions) == 0:
                to_drop.append(transition_name)
                continue
            concatenated_transitions[metric_name][transition_name] = np.vstack(c_transitions)
        for d in to_drop:
            concatenated_transitions[metric_name].pop(d)
            
    concatenated_sessions['at_transitions'] = concatenated_transitions

    return concatenated_sessions

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
    """

    session_list = load.session_list()
    all_sessions = {}
    errors = []
    for p in tqdm(session_list.Path):
        try:
            print(p)
            c_session, c_metrics = process_session(local_path=p, 
                                                   save=save, 
                                                   force=force,
                                                   **kwargs)
            all_sessions[c_session['session_name']] = c_metrics
        except:
            errors.append(p)
            print(f'{p} not taken care of because bug')

    merged = merge_all_sessions(all_sessions)
    io.save_shelve('processed_data/network_metrics',{'merged_sessions':merged})
    return merged,errors

def save_data(session: Dict, 
              metrics:Dict,
              params: Dict[str, Dict]) -> None:
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
    

    d = {'unique_sessions': {session['session_name']: {'session':session,
                                                       'metrics':metrics,
                                                       'params': params}}}

    io.save_shelve('processed_data/network_metrics', dict=d, params=params)

if __name__ == '__main__':

    save = True
    force = True

    df,errors = process_all_sessions(save = save,
                              force = force,
                              params = network_metrics_params,
                              min_durations = min_durations,
                              nbins = states_nbins)
    # sess,m = process_session(local_path = 'Rat08/Rat08-20130713',
    #                          save = save,
    #                          force = force,
    #                          params = network_metrics_params,
    #                          min_durations = min_durations,
    #                          nbins = states_nbins)

