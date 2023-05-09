from settings import upath

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from bk import load
from bk import stats
from bk import compute
from bk import plot
from bk import io
from bk import misc

import neuroseries as nts

from tqdm import tqdm
from functools import reduce
import shelve

from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Sequence
from numpy.typing import ArrayLike

def compute_eib(neurons:ArrayLike,metadata:pd.DataFrame,stru:str,binSize:float)->ArrayLike:
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
    t,bin_matrix = compute.binSpikes(neurons,binSize)
    f_bin_matrix_pyr,_ = misc.filter_neurons(bin_matrix,metadata,stru,'Pyr',True)
    f_bin_matrix_int,_ = misc.filter_neurons(bin_matrix,metadata,stru,'Int',True)
    
    m_pyr = np.nanmean(f_bin_matrix_pyr,0)
    m_int = np.nanmean(f_bin_matrix_int,0)

    eib = m_pyr / (m_pyr+m_int)
    return eib

def compute_cv(neurons:ArrayLike,metadata:pd.DataFrame,stru:str,binSize:float)->ArrayLike:
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

    t,bin_matrix = compute.binSpikes(neurons,binSize)
    f_bin_matrix_pyr,_ = misc.filter_neurons(bin_matrix,metadata,stru,'Pyr',True)

    return np.std(f_bin_matrix_pyr,0) / np.nanmean(f_bin_matrix_pyr,0)

def compute_sync_moving_windows(neurons:ArrayLike,metadata:pd.DataFrame,stru:str,binSize:float,winSize:float,step:float)->ArrayLike:
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
        Size of the window to compute sync
    step : float
        step of the sliding window. If step = winSize there is no overlap

    Returns
    -------
    ArrayLike
        1d vector with sync for the whole session
    """
    
    
    winSize = int(winSize/binSize) #Convert second to idx
    step = int(step / binSize)

    t,bin_matrix = compute.binSpikes(neurons,binSize)
    f_bin_matrix_pyr,_ = misc.filter_neurons(bin_matrix,metadata,stru,'Pyr',True)

    window_strides = np.lib.stride_tricks.sliding_window_view(f_bin_matrix_pyr,winSize,axis = 1)
    window_strides = window_strides[:,::step,:].transpose((1,0,2))
    sync = np.array([compute_sync(win) for win in window_strides])

    return sync

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

def process_session(base_folder: Union[Path, str] = upath['base_folder'],
                    local_path: Union[Path, str] = upath['example_session'],
                    stru:str = 'BLA',
                    min_durations: Dict[str, int] = None,
                    save: bool = False,
                    params = None,
                    force: bool = False) -> pd.DataFrame:
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
    print(save)
    if not force:
        data = io.load_shelve('processed_data/network_metrics')
        if ('unique_sessions' in data) and (session['session_name'] in data['unique_sessions']):
            c_data = data['unique_sessions'][session['session_name']]
            return c_data['session'],c_data['eib'],c_data['cv'],c_data['sync']
    
    
    states = load.sleep_scoring(session)
    neurons, metadata = load.spikes(session)

    eib = compute_eib(neurons,metadata,stru,binSize = params['eib']['binSize'])
    cv = compute_cv(neurons,metadata,stru,binSize = params['cv']['binSize'])
    
    
    
    sync = compute_sync_moving_windows(neurons = neurons,
                                       metadata = metadata,
                                       stru = stru,
                                       binSize = params['sync']['binSize'],
                                       winSize = params['sync']['winSize'],
                                       step = params['sync']['step'])

    if save:
        save_data(session,eib,cv,sync,params)

    return session,eib,cv,sync

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
    for p in tqdm(session_list.Path):
        try:
            print(p)
            c_session, c_eib, c_cv, c_sync = process_session(local_path=p, 
                                                             save=save, 
                                                             force=force,
                                                             **kwargs)
        except:
            print(f'{p} not taken care of because bug')

    return all_sessions

def save_data(session: Dict, 
              eib:ArrayLike, 
              cv:ArrayLike,
              sync:ArrayLike, 
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
                                                       'eib': eib,
                                                       'cv': cv,
                                                       'sync':sync,
                                                       'params': params}}}

    io.save_shelve('processed_data/network_metrics', dict=d, params=params)



if __name__ == '__main__':

    save = False
    force = False
    
    params = {'eib':{
                    'binSize':10},
              'cv':{
                    'binSize':5},
              'sync':{
                    'binSize':0.1,
                    'winSize':10,
                    'step':1}}


    # session = load.session()
    # neurons, metadata = load.spikes(session)
    # t,bin_matrix = compute.binSpikes(neurons)


    # eib,cv,sync = process_session(params = params,save = True,force = True)
    process_all_sessions(save = True,force = True,params = params)
    # plt.ion()
    # fig,ax = plt.subplots(3,1)

    # for i,m in enumerate((eib,cv,sync)):
    #     ax[i].plot(m)
    