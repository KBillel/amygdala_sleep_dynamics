from pathlib import Path
from getpass import getuser
users_paths = {'billel':
               {'base_folder': Path("/mnt/electrophy/Gabrielle/GG-Dataset-Light/"),
                # {'base_folder':Path("/media/billel/My Passeport/Gabrielle/GG-Dataset"),
                'example_session': Path('Rat08/Rat08-20130713')},
               'remi': {'base_folder': Path('.'),
                        'example_session': Path('.')}}
username = getuser()

upath = users_paths[username]

colors = {'NREM': '#808080ff',
          'REM': '#ffa500ff',
          'WAKE_HOMECAGE': '#008bc8ff',
          'extended_sleep': '#ffeeaaff',

          'REM_ON':'#ffa500ff',
          'REM_OFF':'#808080ff',
          'Unknown':'#CCCCFF',

          'BLA': {'BLA': '#85ad8bff',
                  'VH': '#527a58ff',
                  'H': '#67986eff',
                  'M': '#85ad8bff',
                  'L': '#a4c1a9ff',
                  'VL': '#c2d6c5ff'},

          'Hpc': {'Hpc': '#0000FF',
                  'VH': '#060066ff',
                  'H': '#090099ff',
                  'M': '#0c00ccff',
                  'L': '#0f00ffff',
                  'VL': '#3f33ffff'},

          'Pir': {'Pir': '#FF00FF',
                  'VH': '#650065ff',
                  'H': '#990099ff',
                  'M': '#cc00ccff',
                  'L': '#ff00ffff',
                  'VL': '#ff33ffff',
          },
        0:'#808080ff'}

min_durations = {
    'NREM': 200,
    'REM': 50,
    'WAKE_HOMECAGE': 200}

states_nbins = {
    'NREM': 30,
    'REM': 12,
    'WAKE_HOMECAGE': 30,
    'DROWSY': 1,
    'extended_sleep':90}

network_metrics_params = {
    'eib': {
        'binSize': 10},
    'cv': {
        'binSize': 10},
    'sync': {
        'binSize': 0.1,
        'winSize': 10,
        'step': 1}}


oscillations_bands = {
    'delta':[0.1,4],
    'theta':[4,12],
    'low_gamma':[20,40],
    'mid_gamma':[40,80],
    'high_gamma':[80,100]
}
extended_params = {'sleep': {'sleep_th': 60*30,
                        'wake_th': 60,
                        'sub_states': ['NREM', 'REM']},
              'wake': {'sleep_th': 60,
                       'wake_th': 60*30,
                       'sub_states': ['WAKE_HOMECAGE']}}