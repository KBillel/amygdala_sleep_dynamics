import matplotlib.pyplot as plt
import numpy as np

from bk import io
from bk import plot
from settings import colors

if __name__ == '__main__':
    
    plt.ion()
    fig,ax = plt.subplots(1,2,figsize = (12,8))
    extended = io.load_shelve('processed_data/binned_fr_extended')
    c = {'sleep':0,
         'wake':0}
    for id_sess,data in extended['unique_sessions'].items():
        states = data['session']['states']
        all_extended = data['session']['extended_states']
        
        if np.sum(data['metadata'].Region == 'BLA') == 0: continue
        # for s in states:
        #     states[s] = states[s].drop('state',axis = 1)
        for state,extended in all_extended.items():
            if state == 'wake':
                z = 0
            else:
                z = 1
            if len(extended) == 0: continue
            for i,current_extended in extended.iterrows():
                current_states = {}
                for s in states:
                    current_states[s] = states[s].intersect(current_extended)
                    current_states[s] = current_states[s]-current_extended.start
                for s in ['NREM','REM','WAKE_HOMECAGE','DROWSY']:
                    for i,inter in current_states[s].iterrows():
                        inter = inter / 1_000_000
                        duration = inter.end - inter.start
                        ax[z].hlines(c[state],inter.start,inter.end,colors=colors[s],linewidth = 2)
                c[state] += 1

        
    # plt.ylim(0,c+1)
    for axe in ax: 
        axe.set_xlim(0,6000)
        plot.clean_axes(axe)
    ax[0].set_ylabel('EWE #')
    ax[1].set_ylabel('ESE #')
    ax[0].set_xlabel('Times (s)')
    ax[1].set_xlabel('Times (s)')
    ax[0].legend(['NREM','REM','WAKE_HOMECAGE','DROWSY'])

    fig.savefig('output.png')
    fig.savefig('plots/figures/ESE_EWE.svg')


# FOR DEBUG
            # if c== 13:
            #     fig,ax = plt.subplots(2,1,sharex=True)
            #     for s in ['NREM','REM','WAKE_HOMECAGE']:
            #         plot.intervals(states[s],col = colors[s],ax=ax[0])
            #     plot.intervals(extended,'r
            # ',ax= ax[1])
            #     fig.savefig('output1.png')
