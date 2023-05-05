from pathlib import Path
from getpass import getuser
users_paths = {'billel':
               {'base_folder':Path("/mnt/electrophy/Gabrielle/GG-Dataset-Light/"),
                # {'base_folder':Path("/media/billel/My Passeport/Gabrielle/GG-Dataset"),
                'example_session':Path('Rat08/Rat08-20130713')},
               'remi': {'base_folder': Path('.'),
                        'example_session': Path('.')}}
username = getuser()

upath = users_paths[username]

colors = {'NREM': '#808080ff',
          'REM': '#ffa500ff',
          'WAKE_HOMECAGE': '#008bc8ff',

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
                  'VL': '#ff33ffff'}

          }
