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
import scipy.signal as sig


def population_rates(neurons, binSize=0.010, start=0, end=None):
    t, binned = compute.binSpikes(neurons, binSize, start, end)
    pop = binned.sum(0)
    return t, pop


def wvfilt(signal, mfreq, sr=1250.):

    # computing the length of the wavelet for the desired frequency
    wavelen = int(np.round(10.*sr/mfreq))

    # constructs morlet wavelet with given parameters
    wave = sig.morlet(wavelen, w=5, s=1, complete=True)

    # cutting borders
    cumulativeEnvelope = np.cumsum(np.abs(wave))/np.sum(np.abs(wave))
    Cut1 = next(i for (i, val) in enumerate(
        cumulativeEnvelope[::-1]) if val <= (1./2000))
    Cut2 = Cut1
    Cut1 = len(cumulativeEnvelope)-Cut1-1
    wave = wave[range(Cut1, Cut2)]

    # normalizes wavelet energy
    wave = wave/(.5*sum(abs(wave)))

    if (len(wave)) > len(signal):
        print('ERROR: input signal needs at least '+str(len(wave)) +
              ' time points for '+str(mfreq) +
              'Hz-wavelet convolution')
        return None

    # convolving signal with wavelet
    fsignal = np.convolve(signal, wave, 'same')
    return fsignal


def wvSpect(signal, freqs, tfrout=False, runSpectrogram=True, runPhases=False, s=1, w=5, sr=1250.):

    freqs = np.array(freqs).reshape(-1, 1)
    tfr = np.zeros((np.size(freqs), len(signal)), dtype=complex)
    for (fi, f) in enumerate(freqs):
        f = float(f)
        tfr[fi, :] = wvfilt(signal, f, sr)

    if runSpectrogram:
        output = (np.abs(tfr).squeeze(),)
    if tfrout:
        output += (tfr.squeeze(),)
    if runPhases:
        output += (np.angle(tfr).squeeze(),)

    return output[0]


def wavelet_spectrogram(lfp, fmin, fmax, nfreq):
    f_wv = pow(2, np.linspace(np.log2(fmin), np.log2(fmax), nfreq))
    t = lfp.index.values
    output = wvSpect(lfp.values, f_wv)

    return f_wv, t, output


def save_data(session, metadata, neurons, lfp, spec, pop_rates, params, label):
    """
    Save data of :py:func:'process_session' in a shelve

    Parameters
    ----------
    session : dict

    """

    d = {label: {'session': session,
                 'metadata': metadata,
                 'neurons': neurons,
                 'lfp': lfp,
                 'spec': spec,
                 'pop_rates': pop_rates,
                 'params': params}}

    io.save_shelve('processed_data/examples', dict=d, params=params)


def process_session(base_folder: Union[Path, str] = upath['base_folder'],
                    local_path: Union[Path, str] = upath['example_session'],
                    binSize=0.1,
                    start=0,
                    stop=5,
                    chan=None,
                    label=None,
                    save: bool = False) -> Tuple[Dict, pd.DataFrame, Dict[str, Dict], pd.DataFrame]:

    # FIXME -> ASK VITOR FOR PUBLICATION OF CODE
    params = {'path': local_path.as_posix(),
              'binSize': binSize,
              'start': start,
              'stop': stop,
              'chan': chan}
    inter = nts.IntervalSet(start, stop, time_units='s')

    session = load.session(base_folder=base_folder, local_path=local_path)
    states = load.sleep_scoring(session, drop_short_intervals=None)

    lfp = load.lfp(session, chan, start, stop)
    neurons, metadata = load.spikes(session)

    for n in neurons:
        n = n.restrict(inter)

    f_spec, t_spec, Sxx = wavelet_spectrogram(lfp.as_units("s"), 1.5, 100,200)
    print(Sxx.shape)
    t_fr, fr = population_rates(neurons, start=start, end=stop)

    if save:
        spec = (f_spec, t_spec, Sxx)
        pop_rates = (t_fr, fr)
        save_data(session, metadata, neurons, lfp,
                  spec, pop_rates, params, label=label)


if __name__ == "__main__":
    process_session(chan=123, start=0, stop=5, 
                    label='REM_Example', save=True)
    
    process_session(chan=123, start=1000, stop=1005,
                    label='NREM_Example', save=True)
