import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from scipy.stats import linregress, kruskal
from statannotations.Annotator import Annotator
import neuroseries as nts
from bk import io
from bk import plot
from bk.stats import from_statannon, from_scipy
from plots.plot_transitions import plot_activity_at_transitions
from settings import colors

import matplotlib.gridspec as gridspec

def plot_raster(neurons, metadata, region, intervals, ax):
    t_start = intervals.as_units("s").start.values[0]
    t_end = intervals.as_units("s").end.values[0]
    for i, n in enumerate(
        neurons[(metadata.Region == region) & (metadata.Type == "Pyr")]
    ):
        t = n.restrict(intervals).as_units("s").index
        ax.plot(t, t - t + i, "g|")

    for i, n in enumerate(
        neurons[(metadata.Region == region) & (metadata.Type == "Int")],
        len(neurons[(metadata.Region == region) & (metadata.Type == "Pyr")]),
    ):
        t = n.restrict(intervals).as_units("s").index
        ax.plot(t, t - t + i, "k|")
    ax.set_xlim(t_start, t_end)

def plot_population_rates(t_pop, p, intervals, ax):
    t_start = intervals.as_units("s").start.values[0]
    t_end = intervals.as_units("s").end.values[0]

    ax.bar(t_pop, p, t_pop[1] - t_pop[0],color= 'g')
    ax.set_xlim(t_start, t_end)

def plot_example(data,gs):
    
    interval = nts.IntervalSet(data['params']['start'],data['params']['stop'],time_units='s')
    start = interval.as_units('s').start.values[0]
    stop = interval.as_units('s').end.values[0]
    gs00 = gridspec.GridSpecFromSubplotSpec(4,1,gs)
    
    t_spec = data['spec'][1]
    f_spec = data['spec'][0]
    Sxx = data['spec'][2]

    lfp = fig.add_subplot(gs00[0,:])
    spec = fig.add_subplot(gs00[1,:])
    spikes = fig.add_subplot(gs00[2,:])
    pop_rates = fig.add_subplot(gs00[3,:])
    
    axes = [lfp,spec,spikes,pop_rates]

    lfp.plot(data['lfp'].as_units('s'),'g')
    spec.pcolormesh(t_spec,f_spec,Sxx,shading="gouraud", vmin=0, vmax=250, rasterized=True)
    plot_raster(data['neurons'],data['metadata'],'BLA',interval,spikes)
    
    plot_population_rates(data['pop_rates'][0],data['pop_rates'][1],
                          interval,pop_rates)

    for ax in axes:
        plot.full_clean_ax(ax)
    
    pop_rates.spines['bottom'].set_visible(True)
    pop_rates.set_xticks(np.arange(start,stop+1,1),np.arange(0,6,1))
    pop_rates.xaxis.set_ticks_position('bottom')
    pop_rates.set_xlabel('Time (s)')


    spec.spines['left'].set_visible(True)
    spec.set_yticks(np.arange(0,101,25),np.arange(0,101,25))
    spec.yaxis.set_ticks_position('left')
    spec.set_ylabel('Frenquency (Hz)')




if __name__ == '__main__':

    
    data = io.load_shelve('processed_data/examples')



    fig = plt.figure()
    gs0 = gridspec.GridSpec(2,3)

    plot_example(data['NREM_Example'],gs0[0,0])
    plot_example(data['REM_Example'],gs0[0,1])
    plot_example(data['REM_Example'],gs0[0,2])

    plt.ion()
    plt.show()

