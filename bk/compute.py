from dataclasses import replace
import numpy as np
import neuroseries as nts
from tqdm import tqdm
import os
import scipy.stats
import pandas as pd
import itertools as it

from typing import Union, Optional,Tuple, Dict, Sequence
from numpy.typing import ArrayLike


def speed(pos,value_gaussian_filter, columns_to_drop=None):
    
    body = []
    for i in pos:
        body.append(i[0])
    body = np.unique(body)
    
    all_speed = np.empty((len(pos)-1,5))
    i = 0
    for b in body:
        x_speed = np.diff(pos.as_units('s')[b]['x'])/np.diff(pos.as_units('s').index)
        y_speed = np.diff(pos.as_units('s')[b]['y'])/np.diff(pos.as_units('s').index)
    
        v = np.sqrt(x_speed**2 + y_speed**2)
        all_speed[:,i] = v
        i +=1
    all_speed = scipy.ndimage.gaussian_filter1d(all_speed,value_gaussian_filter,axis=0)
    all_speed = nts.TsdFrame(t = pos.index.values[:-1],d = all_speed,columns = body)
    if columns_to_drop != None: all_speed = all_speed.drop(columns=columns_to_drop)
    
    return all_speed

def binSpikes(neurons,binSize = 0.025,start = 0,stop = None,nbins = None,fast = False, centered = True,as_Tsd = False):
    '''
        Bin neuronal spikes with difine binSize.
        If no start/stop provided will run trought all the data
        
        If fast, will assume that two spikes cannot happen in the same bin. 
        
        If centered will return the center of each bin. Otherwise will return edges
        I think that fast is not compatible with centered because already centering.
    '''
    if binSize is not None:
        if binSize < 0.025 and not fast: print(f"You are using {binSize} ms bins with the function fast off. Consider using \"Fast = True\" in order to speed up the computations")
    if stop is None:
        stop = np.max([neuron.as_units('s').index[-1] for neuron in neurons if any(neuron.index)])
    
    bins = np.arange(start,stop,binSize)
    if nbins is not None: bins = np.linspace(start,stop,nbins+1) # IF NUMBER OF BINS IS USED THIS WILL OVERWRITE binSize    
    
    if not fast:
        binned = np.empty((len(neurons),len(bins)-1),dtype = 'int32')
        for i,neuron in enumerate(neurons):
            binned[i],b = np.histogram(neuron.as_units('s').index,bins = bins,range = [start,stop])
    elif fast:
        centered = False
        binned = np.zeros((len(neurons),len(bins)),dtype = np.bool)
        b = bins
        for i,neuron in enumerate(neurons):
            spike_bin = np.unique((neuron.times(units = 's')/binSize).astype(np.int))
            binned[i,spike_bin] = 1
        

    if centered:
        b = np.convolve(b,[.5,.5],'same')[1::]
        b = np.round(b,6)
    if as_Tsd:
        return nts.TsdFrame(b,binned.T,time_units='s')
    return b,binned

def bin_by_intervals(neurons, intervals,as_Tsd = False):
    bins = np.sort(np.concatenate(
        (intervals.as_units('s').start, intervals.as_units('s').end)))

    binned = np.empty((len(neurons), len(bins)-1), dtype='int32')
    for i, neuron in enumerate(neurons):
        binned[i], b = np.histogram(neuron.as_units('s').index, bins=bins)
    t = np.mean((intervals.as_units('s').start,
                intervals.as_units('s').end), 0)

    if as_Tsd:
        return nts.TsdFrame(t,binned[:,::2].T,time_units='s')
    return t, binned[:, ::2]

def jitter_spikes(neuron,tmax,time_units = 'ms'):
    np.random.seed()
    jit = (np.random.rand(len(neuron)) - 0.5)*tmax*2
    new_spikes = neuron.times(time_units)+jit
    return nts.Ts(np.sort(new_spikes),time_units)

def jittered_ppc(neuron,phases,jitter_max,n_spikes,i):
    j_s = jitter_spikes(neuron,jitter_max)
    return ppc(j_s,phases,n_spikes)


def fr(neuron,state):
    return len(neuron.restrict(state))/state.tot_length('s')

def transitions_times(states,epsilon = 1,verbose = False):
    '''
        states : dict of nts.Interval_Set
        
        This function compute transition in between Intervals in a dict.
        It returns a new dict with intervals and when the transition occurs
        
        epsilon : tolerance time delay between state
        
        This function does NOT WORK for triple transitions (ex : sws/rem/sws) ... 
        
    '''
    
    import itertools
    
    empty_state = []
    for state in states:
        if len(states[state]) == 0:
            empty_state.append(state)
            continue
        states[state] = states[state].drop_short_intervals(1)
    
    
    for i in empty_state: del states[i]
        
    transitions_intervals = {}
    transitions_timing = {}
    
    for items in itertools.permutations(states.keys(),2):
#         states[items[0]] = states[items[0]].drop_short_intervals(1)
#         states[items[1]] = states[items[1]].drop_short_intervals(1)
        
        if verbose: print('Looking at transition from',items[0],' to ',items[1])
        end = nts.Ts(np.array(states[items[0]].end + (epsilon * 1_000_000)+1))
        in_next_epoch = states[items[1]].in_interval(end)
        
        transitions_intervals.update({items:[]})
        transitions_timing.update({items:[]})

        for n,t in enumerate(in_next_epoch):
            if np.isnan(t): continue            
            start = states[items[0]].iloc[n].start
            trans = int(np.mean([states[items[0]].iloc[n].end,states[items[1]].iloc[int(t)].start]))
            end  = states[items[1]].iloc[int(t)].end
            transitions_intervals[items].append([start,end])
            transitions_timing[items].append(trans)
        
        if  not transitions_timing[items] == []:      
            transitions_intervals[items] = np.array(transitions_intervals[items])
            transitions_intervals[items] = nts.IntervalSet(transitions_intervals[items][:,0],transitions_intervals[items][:,1],force_no_fix = True)
            
            transitions_timing[items] = nts.Ts(t = np.array(transitions_timing[items]))
    return transitions_intervals,transitions_timing

def nts_smooth(y,m,std):

    if len(y)<m:
        m = len(y)
    g = scipy.signal.gaussian(m,std)
    g = g/g.sum()
    
    if len(y.shape)>1:
        y_ = np.zeros_like(y.values)

        for i in range(y.shape[1]):

            y_[:,i] = np.convolve(y.values[:,i],g,'same')
        y = nts.TsdFrame(y.index.values,y_)
    else:
        conv = np.convolve(y.values,g,'same')
        y = nts.Tsd(y.index.values,conv)
    return y

def smooth(y,m,std):
    g = scipy.signal.gaussian(m,std)
    g = g/g.sum() 
    y = np.convolve(y,g,'same')

    return y

def nts_zscore(tsd,axis = 0):
        
    t = tsd.index.values
    if len(tsd.shape) > 1: 
        z_tsd = scipy.stats.zscore(tsd.values.astype(np.float32),axis)
        return nts.TsdFrame(t,z_tsd)
    else:
        z_tsd = scipy.stats.zscore(tsd.values.astype(np.float32))
        return nts.Tsd(t,z_tsd)

def intervals_exp(force_reload = False, save = False):
    files = os.listdir()
    if ('intervals.npy' in files) and (force_reload == False):
        with open('intervals.npy', 'rb') as f:
            exp = np.load(f)
            shock = np.load(f)
            tone = np.load(f)
            exp = nts.IntervalSet(exp[:,0],exp[:,1],time_units='us')
            shock = nts.IntervalSet(shock[:,0],shock[:,1],time_units='us')
            tone = nts.IntervalSet(tone[:,0],tone[:,1],time_units='us')
            return (exp, shock, tone)
        
    exp = tone_intervals(bk.load.digitalin('digitalin.dat',1))
    shock = tone_intervals(bk.load.digitalin('digitalin.dat',2))
    tone = tone_intervals(bk.load.digitalin('digitalin.dat',3))
    
    if save:
        with open('intervals.npy', 'wb') as f:
            np.save(f, exp)
            np.save(f, shock)
            np.save(f, tone)
    
    return (exp, shock, tone)

def psth(neurons,stimulus,binSize,win,average = True):
    if isinstance(neurons,nts.time_series.Tsd): 
        neurons = np.array(neurons,'object')
    winLen = int((win[1] - win[0])/binSize)
    window = np.arange(winLen,dtype = int)-int(winLen/2)
    stim_bin = (stimulus/binSize).astype('int')
    t,binned = binSpikes(neurons,binSize,start = 0, stop = stimulus[-1]+win[-1])
    psth = np.empty((stimulus.size,len(neurons),winLen))
    
    for i,t in tqdm(enumerate(stim_bin)):
        psth[i] = binned[:,t+window]
    if average:    
        psth = np.mean(psth,0).T
    t = window*binSize
    return t,psth

def crosscorrelogram(neurons,binSize,win):
    if isinstance(neurons,nts.time_series.Tsd): 
        neurons = np.array(neurons,'object')
    winLen = int((win[1] - win[0])/binSize)
    window = np.arange(winLen,dtype = int)-int(winLen/2)
    crosscorr = np.empty((winLen,len(neurons),len(neurons)),dtype = 'int16')
    last_spike = np.max([n.as_units('s').index[-1] for n in neurons])
    t,binned = binSpikes(neurons,binSize,start = 0, stop = last_spike+win[-1]+1)

    for i,n in tqdm(enumerate(neurons),total = len(neurons)):
        stimulus = n.as_units('s').index
        stim_bin = (stimulus/binSize).astype('int64')
        psth = np.empty((stimulus.size,len(neurons),winLen),dtype = 'int16')

        for j,t in enumerate(stim_bin):
            psth[j] = binned[:,t+window]
#             psth[j][:,window == 0] -= 1

        psth = np.sum(psth,0).T
        crosscorr[:,i] = psth
        t = window*binSize
        
    return t,crosscorr

def toIntervals(t,is_in,time_units = 'us'):
    
    '''
    Author : BK (Inspired Michael Zugaro FMA Toolbox)
    This function convert logical vector to interval.
    '''
    
    if is_in[-1] == 1: is_in = np.append(is_in,0)
    d_is_in = np.diff(is_in,prepend=0)
    start = np.where(d_is_in == 1)[0]
    end = np.where(d_is_in == -1)[0]-1

    
    return nts.IntervalSet(start = t[start],end = t[end],time_units = time_units).drop_short_intervals(0).reset_index(drop = True)

def transition(states, template, epsilon=0):
    """
    author: BK
    states : dict of nts.Interval_set
    template : list of state.
    epsilon : int, will drop any 
     in which there is an epoch shorter than epsilon 's'
    This function will find transition that match the template 
    """
    if epsilon is list:
        print("eplist")
    long = pd.DataFrame()
    for s, i in states.items():
        i["state"] = s
        long = pd.concat((i, long))
        del i["state"]
    order = np.argsort(long.start)
    long = long.iloc[order]

    transition_times = []
    transition_intervals = []
    for i, s in enumerate(long.state):
        tmp = list(long.state.iloc[i: i + len(template)])
        if tmp == template:
            tmp_transition = long.iloc[i: i + len(template)]
            #             print(d.iloc[i:i+len(template)])
            length = (tmp_transition.end - tmp_transition.start) / 1_000_000
            if np.any(length.values < epsilon):
                continue
            tmp_pre = np.array(tmp_transition.end.iloc[:-1])
            tmp_post = np.array(tmp_transition.start.iloc[1:])
            tmp_times = np.mean([tmp_pre, tmp_post], 0)

            transition_intervals.append(
                [tmp_transition.start.iloc[0], tmp_transition.end.iloc[-1]]
            )
            transition_times.append(tmp_times)

    transition_times = np.array(transition_times)
    transition_intervals = np.array(transition_intervals)
    if len(transition_intervals) == 0: 
        return False,False
    transition_intervals = nts.IntervalSet(
        start=transition_intervals[:, 0],
        end=transition_intervals[:, 1],
        force_no_fix=True,
    )
    return transition_intervals, transition_times


def compute_transition_activity(neurons, intervals, timing, bin_epochs, n_event):

    transition_activity = []
    for event, t in zip(intervals.iloc, timing):  # For each transitions
        if n_event == 2:
            epochs = np.array(
                [(event.start, t[0]), (t[0], event.end)], dtype=np.int64)
        if n_event == 3:
            epochs = np.array(
                [[event.start, t[0]], [t[0], t[1]], [t[1], event.end]])
        epochs = nts.IntervalSet(start=epochs[:, 0], end=epochs[:, 1])
        # Creates intervals for each state of the transitions events.

        #         binned = np.array(shape = (252,np.sum(bin_epochs)))
        #         binned = np.empty(shape = (252,np.sum(bin_epochs),len(intervals)+1))
        binned = np.empty(shape=(len(neurons), 1))
        for i, epoch in enumerate(epochs.as_units("s").iloc):
            start = epoch.start
            end = epoch.end
            nbins = bin_epochs[i]
            _, b = bk.compute.binSpikes(
                neurons, start=start, stop=end, nbins=nbins)
            b = b / ((end - start) / nbins)  # Converting to firing rates
            binned = np.hstack((binned, b))
        binned = binned[:, 1:]
        transition_activity.append(binned)

    transition_activity = np.array(transition_activity)
    transition_activity = np.moveaxis(transition_activity, 0, 2)

    return transition_activity



def mean_resultant_length(angles,weights = None):
    angles = np.exp(1j*angles)
    return np.abs(np.average(angles,weights = weights))

def concentration(angles,weights = None):
    '''
    Compute the kappa parameter Uses the approximation described in "Statistical Analysis of Circular Data" (Fisher, p. 88).
    Translated from MATLAB fmatoolbox.sourceforge.net/Contents/FMAToolbox/General/Concentration.html
    angles : radian

    Copyright (C) 2004-2011 by Michaël Zugaro
    Copyright (C) 2021 by Billel KHOUADER

    '''
    n = len(angles)
    r = mean_resultant_length(angles,weights= weights)

    if r < 0.53:
        k = 2 * r + r**3 + 5*r**(5/6)
    elif r < 0.85:
        k = -0.4 + 1.39 * r + 0.43 / (1-r)
    else:
        k = 1/(r**3 - 4 * r**2 + 3*r)

    # Correction for very small samples
    if weights is None:
        if n <= 15:
            if k < 2:
                k = np.max([(k-2)/(n*k), 0])
            else:
                k = (n-1)**3 * k / (n**3+n)

    return k

def downsample_spikes(neuron,n):
    return nts.Ts(np.sort(np.random.choice(neuron.times(),n,replace = False)))

def ppc(neuron, phase, n = None,n_shuffles = None):
    if (n is not None) and (len(neuron)>n):
        neuron = nts.Ts(np.sort(np.random.choice(neuron.times(),n,replace = False)))
    
    neuron_phase = phase.realign(neuron).values

    pcc = neuron_phase[None, :] - neuron_phase[:, None]
    pcc[np.diag_indices_from(pcc)] = np.nan
    pcc = np.cos(pcc)
    return np.nanmean(pcc, 0).mean()

def interval_rates(intervals, binSize, time_units='s'):
    '''
    
    This function return rates of occurence of an intervals as a Tsd.
    
    '''
    bins = np.linspace(min(intervals.as_units(time_units).start), max(intervals.as_units(time_units).end), binSize)
    interval_rates = []
    for i in range(len(bins)-1):
        inter = nts.IntervalSet(bins[i], bins[i+1], time_units=time_units)
        interval_rates.append(len(inter.intersect(intervals)))

    bins_center = np.convolve(bins, [0.5, 0.5], 'same')[1::]

    return nts.Tsd(bins_center, interval_rates, time_units)

def intervals_to_list_of_intervals(intervals):
    interval_list = []
    for start,end in intervals.iloc:
        interval_list.append(nts.IntervalSet(start,end))
    return interval_list
    
def cumsum_ditribution(x,nBins,density = False):
    counts, bins = np.histogram(x,nBins,density = density)
    counts = counts / np.sum(counts)

    bins = np.convolve(bins,[0.5,0.5],'same')[1::]

    return bins,np.cumsum(counts)


def extended(states:Dict[str,nts.IntervalSet], state:str='sleep', sleep_th:int=60*30, wake_th:int=60)->nts.IntervalSet:
    """
    Comptuted extended sleep or extended wake

    Parameters
    ----------
    states : Dict[nts.IntervalSet]
        states as given by :py:func:`load.intervals` 
    state : str, optional
        string 'WAKE' or 'SLEEP', by default 'sleep'
    sleep_th : int, optional
        Amount of minimum sleep to be considered extended or maximum of sleep to not disturb WAKE, by default 60*30
    wake_th : int, optional
        Amount of wake sleep to be considered extended or maximum of WAKE to not disturb sleep, by default 60
, by default 60

    Returns
    -------
    nts.IntervalSet
        extended intervals of a state
    """
    if state.lower() == 'sleep':
        extended = states['NREM'].union(
            states['REM']).merge_close_intervals(wake_th, time_units='s')
        extended = extended[extended.duration(
            time_units='s') > sleep_th].reset_index(drop=True)

    elif state.lower() == 'wake':
        extended = states['WAKE_HOMECAGE'].merge_close_intervals(
            sleep_th, time_units='s')
        extended = extended[extended.duration(
            time_units='s') > wake_th].reset_index(drop=True)
    return extended