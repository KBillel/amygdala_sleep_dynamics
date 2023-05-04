import pandas as pd
import numpy as np

def states_to_longstates(states):
    '''
    This function return the long version of the states variables given
    '''
    long = pd.DataFrame()
    for s,i in states.items():
        i['state'] = s
        long = pd.concat((i,long))
        del i['state']
    order = np.argsort(long.start)
    long = long.iloc[order]
    
    return long

def discard_border(state, t):
    state.start = state.start + (t * 1_000_000)
    state.end = state.end - (t * 1_000_000)

def deep_update(source, overrides):
    """
    from : https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


def filter_neurons(activity,metadata,stru,types,finite):
    mask_stru = metadata.Region == stru
    mask_types = metadata.Type == types
    mask = mask_stru & mask_types
    if finite:
        mask_finite = np.all(np.isfinite(activity),1)
        mask = mask & mask_finite
    return activity[mask,:].copy(), metadata[mask].copy()