#%config InlineBackend.figure_format = 'retina'
import IPython
isColab = 'google.colab' in str(IPython.get_ipython())

from termcolor import colored
import platform
HOSTNAME = platform.node().split('.')[0]

import os
HOME = os.environ['HOME']

#try:
#    import ipynbname
#except ImportError:
#    !pip install ipynbname > /dev/null
import ipynbname
FILEPATH = str(ipynbname.path()).replace(HOME+'/','')

import pwd
USER=pwd.getpwuid(os.geteuid())[0]

from datetime import date
TODAY=date.today()

import torch
TORCH_VERSION = torch.__version__

color = 'green'
print('日付:',colored(f'{TODAY}', color=color, attrs=['bold']))
print('HOSTNAME:',colored(f'{HOSTNAME}', color=color, attrs=['bold']))
print('ユーザ名:',colored(f'{USER}', color=color, attrs=['bold']))
print('HOME:',colored(f'{HOME}', color=color,attrs=['bold']))
print('ファイル名:',colored(f'{FILEPATH}', color=color, attrs=['bold']))
print('torch.__version__:',colored(f'{TORCH_VERSION}', color=color, attrs=['bold']))
