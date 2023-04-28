import os
import pandas as pd
import json
import shelve

from pathlib import Path
def save_dataframe(df,name):
    df.to_csv(f'Analysis/{name}.csv')

def load_dataframe(name):
    print(f'Analysis/{name}.csv')
    if os.path.exists(f'Analysis/{name}.csv'):
        return pd.read_csv(f'Analysis/{name}.csv')
    else:
        print('Could not find the requested analysis file')
        return False

def save_shelve(file_path,dict,params = None,**kwargs):
    import numpy as np
    file_path = Path(file_path)
    if params is not None:
        with open(file_path.with_suffix('.json'),'w') as jf:
            json.dump(params,jf,indent = 4)
    
    
    with shelve.open(file_path.as_posix(),flag='c',**kwargs) as sf:
        prev_dict = {k:v for k,v in sf.items()}
    
    prev_dict.update(dict)
    
    with shelve.open(file_path.as_posix(),flag='n',**kwargs) as sf:
        for k,v in prev_dict.items(): sf[k] = v

def load_shelve(file_path,**kwargs):
    file_path = Path(file_path)
    with shelve.open(file_path.as_posix(),**kwargs) as sf:
        return {k:v for k,v in sf.items()}