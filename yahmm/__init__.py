#!/usr/bin/env python2.7
# yahmm.py: Yet Another Hidden Markov Model library
# Contact: Jacob Schreiber ( jmschreiber91@gmail.com )
#          Adam Novak ( anovak1@ucsc.edu )

"""
For detailed documentation and examples, see the README.
"""

import numpy as np
import pyximport
pyximport.install( setup_args={ 'include_dirs':np.get_include(),
						'options': {'build_ext': {'compiler': 'mingw32'}}})
from yahmm import *
