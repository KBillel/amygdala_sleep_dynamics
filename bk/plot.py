import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import neuroseries as nts
import bk.compute
import scipy.stats

def rasterPlot(neurons,window = None,col = 'black',width= 0.5,height = 1,offsets = 1, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    if window is None:
        last_spike = np.array([n.as_units('s').index.values[-1] for n in neurons])
        window = np.array([[0,np.max(last_spike)]])
            
    window = np.atleast_2d(np.array(window))
    window = nts.IntervalSet(window[:,0],window[:,1],time_units = 's')

    neurons_np = []
    
    if isinstance(neurons,nts.time_series.Tsd):
        neurons = [neurons]
    for neuron in neurons:
        neurons_np.append(neuron.restrict(window).as_units('s').index)

    neurons_np = np.array(neurons_np,dtype = 'object')
    
    
    ax.eventplot(neurons_np,color = col,linewidth = width,linelengths=height,lineoffsets=offsets)
    ax.set_ylabel('Neurons')
    ax.set_xlabel('Time(s)')
    return ax


def intervals(intervals,col = 'orange',alpha = 0.5,time_units = 's',ymin = 0,ymax = 1,ax = None):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    if not isinstance(intervals, nts.interval_set.IntervalSet):
        print(type(intervals))
        # Fixme: This assumes a dict/dataframe. How useful is it?
        intervals = nts.IntervalSet(intervals['start'],intervals['end'])
    
    for interval in intervals.as_units(time_units).values:
        ax.axvspan(interval[0],interval[1], facecolor=col, alpha=alpha,ymin = ymin, ymax = ymax)
    return ax


def spectrogram(t,f,spec,log = False,ax = None,vmin = None,vmax = None):
    if ax is None:
        fig, ax = plt.subplots(1,1)

    if log:
        spec = np.log(spec)
    ax.pcolormesh(t,f,spec)
    return ax

    
def cumsum_curves(x,nbins,col = 'orange',ax = None, log = False):
    x,y = bk.compute.cumsum_ditribution(x,nbins)

    if ax is None:
        fig,ax = plt.subplots(1,1)

    if log:
        ax.semilogx(x,y,col)
    else:
        ax.plot(x,y,col)
    return ax


def confidence_intervals(x,y,style = None,alpha = 0.5,ax = None):
    # Todo: To remove
    if ax is None:
        fig,ax = plt.subplots(1, 1)
    
    conf = 1.96*scipy.stats.sem(y,0,nan_policy='omit')
    m = np.nanmean(y,0)
    ax.fill_between(x,m+conf,m-conf,color = style,alpha = alpha)
    ax.plot(x,m,c = style)
    return ax


def curve_and_shades(x,y,method = 'std',style = 'orange',ax = None):
    # Fixme: This is very similar to confidence_intervals but the sem calculation is different
    # They should be merged, homogeneized and only one should be used
    if ax is None:
        fig,ax = plt.subplots(1,1)
    if method.lower() == 'std':
        shade = np.nanstd(y,0)
    elif method.lower() == 'sem':
        shade = scipy.stats.sem(y,0,nan_policy='omit')   # * 1.96 ?
     
    m = np.nanmean(y,0)
    ax.plot(x,m,style)
    ax.fill_between(x,m+shade,m-shade,color = style,alpha = 0.2)


def forceAspect(ax,aspect=1):
    #aspect is width/height
    scale_str_y = ax.get_yaxis().get_scale()
    scale_str_x = ax.get_xaxis().get_scale()

    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    
    if scale_str_x == 'log': xmin,xmax = np.log10(xmin),np.log10(xmax)
    if scale_str_y == 'log': ymin,ymax = np.log10(ymin),np.log10(ymax)
    asp = abs((xmax-xmin)/(ymax-ymin))/aspect

    ax.set_aspect(asp)

def clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def set_share_axes(axs, target=None, sharex=False, sharey=False):
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1,:].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:,1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)