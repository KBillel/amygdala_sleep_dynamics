from settings import upath

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from bk import load
from bk import stats
from bk import compute

import neuroseries as nts

from tqdm import tqdm
from functools import reduce

from pathlib import Path
from typing import Union, Optional,Tuple, Dict, Sequence
from numpy.typing import ArrayLike


def process_session(base_folder:Union[Path,str]= upath['base_folder'],local_path:Union[Path,str]=upath['example_session'],discarded_states:Sequence[str] = ('DROWSY','WAKE'))->pd.DataFrame:
    md = load.session(base_folder=base_folder,local_path=local_path)
    discarded_states = set(discarded_states)

    neurons,metadata = load.spikes(md)
    metadata['SessID'] = metadata.index

    id_columns = list(metadata.columns)
    states = load.sleep_scoring(md,discard = discarded_states)
