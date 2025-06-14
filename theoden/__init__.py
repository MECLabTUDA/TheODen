from .common import (
    GlobalContext,
    Transferables,
    Transferable,
    ExecutionResponse,
    MetricResponse,
    ResourceResponse,
)
from . import security as security
from . import common as common
from .resources import data as data
from . import networking as net
from . import datasets as datasets
from . import topology as topology
from . import operations as operations
from . import models as models
from . import watcher as watcher

from .start import start_client, start_server, start_storage

__version__ = "0.2.7"

print(
    f""" _____  _             ___   ____               
|_   _|| |__    ___  / _ \ |  _ \   ___  _ __  
  | |  | '_ \  / _ \| | | || | | | / _ \| '_ \ 
  | |  | | | ||  __/| |_| || |_| ||  __/| | | |
  |_|  |_| |_| \___| \___/ |____/  \___||_| |_|
   The Open Distributed Learning Environment
                     {__version__}\n"""
)
