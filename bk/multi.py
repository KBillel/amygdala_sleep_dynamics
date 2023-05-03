import bk.compute
import numpy as np
import multiprocessing as mp
import itertools as it


def multiprocessing(func, args, workers):
    with mp.Pool(workers) as ex:
        res = ex.starmap(func, args)
    return list(res)


def jittered_ppc(neuron,phases,jitter_max,n_spikes,n_shuffles,n_workers):
    
    shuffles = multiprocessing(bk.compute.jittered_ppc, 
                                    zip(it.repeat(neuron), 
                                        it.repeat(phases), 
                                        it.repeat(jitter_max), 
                                        it.repeat(n_spikes), 
                                        range(n_shuffles)), 
                                        n_workers)
    return np.array(shuffles)


import collections
def deep_update(source, overrides):
    """
    from : https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.iteritems():
        if isinstance(value, collections.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source