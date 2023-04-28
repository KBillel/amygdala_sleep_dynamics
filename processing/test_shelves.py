import shelve
import numpy as np
from bk.io import save_shelve
from bk.load import session_list
dict = {'k':np.random.rand(1000,1000),
        'v':np.random.rand(1000,1000)}

l = session_list()

for p in l.Path:
    x = np.random.rand(np.random.randint(1,500),np.random.randint(1,500))
    save_shelve('test',{p:x})
