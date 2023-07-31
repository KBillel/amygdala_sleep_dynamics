import matplotlib.pyplot as plt

from bk import io
from bk import plot

plt.rcParams['svg.fonttype'] = 'none'
from scipy.stats import zscore

from settings import colors


def plot_oscillations(values,ax = None):
    if ax is None:
        fig,ax = plt.subplots()
    x = range(values.shape[1])
    values = zscore(values,1)
    plot.confidence_intervals(x,values,alpha=0.2,style = colors['BLA']['BLA'],ax = ax)

def plot_all_oscillations(oscillations,ax = None):
    for side,c_side in oscillations.items():
        for i,(band,c_band) in enumerate(c_side.items()):
            if (band == 'delta') or (band == 'theta'): 
                for j,(state,c_state) in enumerate(c_band.items()):
                    plot_oscillations(c_state,ax[i,j])
                    if i==0: ax[i,j].set_title(state)
                    if j==0: ax[i,j].set_ylabel(f'Power-{band}\n(std)')
                    ax[i,j].set_ylim(-2,2)

if __name__ == '__main__':
    fig,ax = plt.subplots(2,3,figsize = (12,8),sharey='row',sharex='col',squeeze=False)

    oscillations = io.load_shelve('processed_data/oscillations')['merged_sessions']
    plot_all_oscillations(oscillations,ax)
    for axe in ax.flatten(): 
        axe.set_xlabel('Time (bins)')
        plot.clean_axes(axe)
    fig.tight_layout()
    fig.savefig('output.png')
    fig.savefig('plots/figures/oscillations.svg')
    fig.show()
