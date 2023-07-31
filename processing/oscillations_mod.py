from settings import upath, states_nbins, min_durations, colors, oscillations_bands_mod

from scipy.signal import spectrogram
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from bk import signal
from bk import load
from bk import stats
from bk import compute
from bk import plot
from bk import io

import neuroseries as nts

from tqdm import tqdm

from pathlib import Path
from typing import Union, Optional, Tuple, Dict, List
from numpy.typing import ArrayLike


def circular_mean(phases,weights):
    phases = np.exp(1j*phases)
    average = np.average(phases,weights = weights)
    return np.angle(average)

def get_best_neighbour_channel(session,shank):
    neighbours_shank = load.shank_neighbours(session,shank)
    if np.isfinite(neighbours_shank['medial']):
        neighbour_channel = load.best_channel(session,neighbours_shank['medial'])
    else:
        neighbour_channel = load.best_channel(session,neighbours_shank['lateral'])
    return neighbour_channel

def compute_phases_lfps(lfps_filt):
    phases_lfps = {}
    for state,lfps in lfps_filt.items():
        phases_lfps[state] = {}
        for band,lfp_filt in lfps.items():

            phases_lfps[state][band] = {} 

            _, phases = signal.hilbert(lfp_filt)


            phases_signal_distrib, _ = np.histogram(phases.values,100,density=True)
            weights = phases_signal_distrib*100 / np.sum(phases_signal_distrib)
            
            phases_lfps[state][band]['values'] = phases
            phases_lfps[state][band]['weights'] = weights

    return phases_lfps


def filter_lfps(lfps,oscillations_bands):
    filt_lfps = {}
    for state,lfp in lfps.items():
        filt_lfps[state] = {}
        for band,(low,high) in oscillations_bands.items():
            filt_lfps[state][band] = signal.passband(lfp,low,high,order = 2)
    return filt_lfps

def load_neighbours_lfps(session,shank,states):
    
    chan = get_best_neighbour_channel(session,shank)
    
    print(f'Loading {chan} from {shank}')

    lfps = {}
    for state,intervals in states.items():
        lfps[state] = load.lfp_in_intervals(session,chan,intervals)
    print('Lfps Loaded. \nProcessing passband and Hilberts')
    return lfps

def compute_mod_features(neuron,phases):
    neuron_phase = phases['values'].realign(neuron)
    phases_distribution, bin_p = np.histogram(neuron_phase.values,100)
    phases_distribution = phases_distribution/phases['weights']
    bin_p = np.convolve(bin_p,[0.5,0.5],'same')[1::]
    if len(neuron) == 0:
        return bin_p, phases_distribution,np.nan,np.nan,np.nan
    mrl = compute.mean_resultant_length(bin_p,phases_distribution)
    pvalue = stats.rayleigh(bin_p,phases_distribution)
    average =  circular_mean(bin_p, phases_distribution)
    
    return bin_p, phases_distribution, average, mrl, pvalue 

def compute_all_modulation_features(neuron,lfps_phases,states):
    features = {}
    for state,lfps_phases in lfps_phases.items():
        features[state] = {}
        for band,phases in lfps_phases.items():
            c_state = states[state]
            bin_p, phases_distributions, average, mrl, pvalue = compute_mod_features(neuron.restrict(c_state),phases)
            
            
            features[state][band] = {'bin':bin_p,
                                     'phases_distributions':phases_distributions,
                                     'average_phase':average,
                                     'mrl':mrl,
                                     'pvalue':pvalue}
    return features

def stack_features(features_bands):
    for state,all_bands_features in features_bands.items():
        for band,c_features in all_bands_features.items():
            for f,data in c_features.items():
                features_bands[state][band][f] = np.vstack(data)
    return features_bands


def merge_features(features):
    concatenated_features = {}
    for i,c_features in enumerate(features):
        for state,c_bands in c_features.items():
            concatenated_state_features = concatenated_features.get(state,{})
            for band,c_data in c_bands.items():
                concatenated_band_features = concatenated_state_features.get(band,{'bin':[],
                                                                                   'phases_distributions':[],
                                                                                   'average_phase':[],
                                                                                    'mrl':[],
                                                                                    'pvalue':[]})
                concatenated_band_features['bin'].append(c_data['bin'])
                concatenated_band_features['phases_distributions'].append(c_data['phases_distributions'])
                concatenated_band_features['average_phase'].append(c_data['average_phase'])
                concatenated_band_features['mrl'].append(c_data['mrl'])
                concatenated_band_features['pvalue'].append(c_data['pvalue'])

                concatenated_state_features[band] = concatenated_band_features
            concatenated_features[state] = concatenated_state_features
    return concatenated_features


def compute_mod(session,neurons,metadata,states,oscillations_bands):
    
    current_shank = 0
    features = []
    for n,(_,m) in tqdm(zip(neurons,metadata.iterrows())):
        if m.Shank != current_shank:
            lfps = load_neighbours_lfps(session,m.Shank,states)
            lfps_filt = filter_lfps(lfps,oscillations_bands)
            lfps_phases = compute_phases_lfps(lfps_filt)
            current_shank = m.Shank
        features.append(compute_all_modulation_features(n,lfps_phases,states))
    features = merge_features(features)
    features = stack_features(features)
    return features

def process_session(base_folder: Union[Path, str] = upath['base_folder'],
                    local_path: Union[Path, str] = upath['example_session'],
                    min_durations: Dict[str, int] = {},
                    discard = ['DROWSY','WAKE'],
                    oscillations_bands: Dict[str, Tuple] = {},
                    save: bool = False,
                    force: bool = True):
    session = load.session(base_folder, local_path)   
    if not force:
        data = io.load_shelve('processed_data/oscillations_modulation')
        if ('unique_sessions' in data) and (session['session_name'] in data['unique_sessions']):
            c_data = data['unique_sessions'][session['session_name']]
            return c_data['session'], c_data['metadata'], c_data['data']

    neurons, metadata = load.spikes(session)
    states = load.sleep_scoring(session,drop_short_intervals=min_durations,discard=discard)
    for s in states:
        states[s] = states[s].drop('state',axis = 1)
    data = compute_mod(session,neurons,metadata,states,oscillations_bands)

    if save:
        save_data(session,metadata, data, oscillations_bands, min_durations)
    return session,metadata,data

def merge_sessions(all_sessions):
    l_metadata = []
    l_features = []
    
    for name, data in all_sessions.items():
        l_metadata.append(data['metadata'])
        l_features.append(data['data'])
    
    l_features = merge_features(l_features)
    s_features = stack_features(l_features)
    metadata = pd.concat(l_metadata,axis = 0)

    merged = {'metadata':metadata,
              'mod_features':s_features}

    return merged
    


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
    errors = []

    for p in tqdm(session_list.Path):
        try:
            c_session, c_metadata, c_data = process_session(local_path=p, 
                                                            save=save, 
                                                            force=force,
                                                            **kwargs)
            all_sessions[c_session['session_name']] = {'session':c_session,
                                                        'metadata': c_metadata,
                                                       'data': c_data,}
        except:
            errors.append(p)
            print(f'{p} not taken care of because bug')
    merged = merge_sessions(all_sessions)
    io.save_shelve('processed_data/oscillations_modulation',{'merged_sessions':merged})


    return merged,errors




def save_data(session: Dict, 
              metadata: pd.DataFrame, 
              data: Dict[str, nts.IntervalSet], 
              oscillations_bands,
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
    
    params = {'oscillations_band': oscillations_bands,
              'min_durations': min_durations}

    d = {'unique_sessions': {session['session_name']: {'session':session,
                                                       'metadata': metadata,
                                                       'data': data,
                                                       'params': params}}}

    io.save_shelve('processed_data/oscillations_modulation', dict=d, params=params)

if __name__ == '__main__':
    # data = process_session(local_path = 'Rat08/Rat08-20130712',
    #                        min_durations=min_durations,
    #                        oscillations_bands=oscillations_bands_mod,
    #                        save = True,
    #                        force = False)

    data = process_all_sessions(min_durations=min_durations,
                                oscillations_bands=oscillations_bands_mod,
                                save = True,
                                force = False)
    # data = io.load_shelve('processed_data/oscillations_modulation')