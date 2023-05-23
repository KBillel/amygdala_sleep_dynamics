from settings import upath, states_nbins, min_durations, colors, oscillations_bands

from scipy.signal import spectrogram
from scipy.stats import zscore
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


from transitions import find_transitions
from network_metrics import make_epochs_average, clean_metrics


def average_by_band(f, Sxx, band, norm=False):
    mask_f = (band[0] < f) & (f < band[1])
    average = np.nanmean(Sxx[mask_f, :], 0)
    if norm:
        return zscore(average)
    return average


def get_power_by_band(session, best_bla_channels, oscillations_bands):
    lfp = {}
    power = {'raw': {},
             'averaged': {}}
    for side, chan in best_bla_channels.items():
        if np.isnan(chan):
            break
        t_lfp, lfp[side] = load.lfp(session, chan, memmap=True)
        f, t_spec, Sxx = spectrogram(
            lfp[side], 1250, nperseg=5000, noverlap=2500)

        power['raw'][side] = {}
        for band, freq in oscillations_bands.items():
            power['raw'][side][band] = (t_spec, average_by_band(f, Sxx, freq))
    return power


def averaged_by_bins(power, states, nbins):
    for side, values in power['raw'].items():
        power['averaged'][side] = make_epochs_average(power['raw'][side], states, nbins=nbins)
    return power


def process_session(base_folder: Union[Path, str] = upath['base_folder'],
                    local_path: Union[Path, str] = upath['example_session'],
                    nbins: Dict[str, int] = {},
                    min_durations: Dict[str, int] = {},
                    oscillations_bands: Dict[str, Tuple] = {},
                    save: bool = False,
                    force: bool = True):
    """
    Process a session with computation relative to oscillations at transitions


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

    """
    session = load.session(base_folder, local_path)
    if not force:
        data = io.load_shelve('processed_data/oscillations')
        if ('unique_sessions' in data) and (session['session_name'] in data['unique_sessions']):
            c_data = data['unique_sessions'][session['session_name']]
            return c_data['session'], c_data['power']

    states = load.sleep_scoring(
        session, discard=['WAKE', 'DROWSY'], drop_short_intervals=min_durations)

    best_bla_channels = load.bla_channels(session)
    power = get_power_by_band(session, best_bla_channels, oscillations_bands)
    power = averaged_by_bins(power, states, nbins)

    if save:
        save_data(session, power, best_bla_channels,
                  nbins, min_durations, oscillations_bands)
    return session, power


def save_data(session, power, best_channels, nbins, min_durations, oscillations_bands):
    params = {'bla_channels': best_channels,
              'nbins': nbins,
              'min_durations': min_durations,
              'oscillations_bands': oscillations_bands}

    d = {'unique_sessions': {session['session_name']: {'session': session,
                                                       'power': power,
                                                       'params': params}}}
    params.pop('bla_channels')
    io.save_shelve('processed_data/oscillations', d, params)


def append_session(concatenated_session, session):
    for side, c_averaged in session['power']['averaged'].items():
        bands = session['power']['raw'][side].keys()
        prev_concat_session = concatenated_session.get(side, {band: {'NREM': [],
                                                                     'REM': [],
                                                                     'WAKE_HOMECAGE': []} for band in bands})
        for band in bands:
            for state in prev_concat_session[band].keys():
                prev_concat_session[band][state].extend(
                    c_averaged[band][c_averaged['state'] == state].values)
        concatenated_session[side] = prev_concat_session

    return concatenated_session


def merge_all_sessions(all_sessions: Dict[str, Dict]) -> Dict[str, Dict]:
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
    for _, c_session in all_sessions.items():
        concatenated_sessions = append_session(
            concatenated_sessions, c_session)

    for average_kind, all_metrics in concatenated_sessions.items():
        all_metrics = clean_metrics(all_metrics)

    return concatenated_sessions


def process_all_sessions(base_folder: Union[Path, str] = upath['base_folder'],
                         save: bool = False,
                         force: bool = False,
                         **kwargs) -> Dict:
    """
    Run :py:func:'process_session' for all session in the dataset

    Parameters
    ----------
    base_folder : Union[Path,str], optional
        Path to the dataset, by default upath['base_folder']
    save : bool
        if true save individual data
    force : bool
        if true will force to recompute instead of loading data

    Returns
    -------
    Dict
        Merged Dict 
    # """

    session_list = load.session_list()
    all_sessions = {}
    for p in tqdm(session_list.Path):
        try:
            c_session, c_power = process_session(local_path=p,
                                                 save=save,
                                                 force=force,
                                                 **kwargs)
            all_sessions[c_session['session_name']] = {'power': c_power}
        except:
            print(f'{p} not taken care of because bug')

    merged = merge_all_sessions(all_sessions)
    io.save_shelve('processed_data/oscillations', {'merged_sessions': merged})

    return merged


if __name__ == '__main__':

    p = process_all_sessions(nbins=states_nbins,
                             min_durations=min_durations,
                             oscillations_bands=oscillations_bands,
                             save=True,
                             force=False)
    # a = process_session(local_path = 'Rat09/Rat09-20140408',force = False)
