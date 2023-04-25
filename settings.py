from pathlib import Path
from getpass import getuser
users_paths = {'billel':
               {'base_folder':Path("/mnt/electrophy/Gabrielle/GG-Dataset-Light/"),
                'example_session':Path('Rat08/Rat08-20130713')}}
username = getuser()

upath = users_paths[username]

colors = {'NREM':'#808080ff',
          'REM':'#ffa500ff',
          'WAKE_HOMECAGE':'#25b262ff'}