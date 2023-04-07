from bk import load
import pandas as pd


def states_fr(base_folder,local_path,discarded_states = ('DROWSY','WAKE')):
    md = load.session(base_folder=base_folder,local_path=local_path)
    neurons,metadata = load.spikes(md)
    
    states = load.intervals(md,'Intervals/sleep_scoring.csv',discarded_states)
    
    fr_states = pd.DataFrame()
    for s,intervals in states.items():
        fr = [len(n.restrict(intervals))/intervals.tot_length(time_units = 's') 
              for n in neurons]
        fr_states[s] = fr
    
    tot_spikes = [len(n) for n in neurons]
    
    fr_states['tot_spikes'] = tot_spikes
    metadata['SessID'] = metadata.index
    
    return pd.concat((metadata,fr_states),axis=1)


if __name__ == "__main__":
    # FIXME : WAKE_HOMECAGE ?
    print(states_fr('/mnt/electrophy/Gabrielle/GG-Dataset-Light/','Rat08/Rat08-20130713'))
    # FIXME : SAVE DATE