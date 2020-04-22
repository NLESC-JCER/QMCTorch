#!/usr/bin/env python
import os
from h5xplorer.h5xplorer import h5xplorer
import context_menu
import qmctorch


base = os.path.dirname(qmctorch.__file__) + "/../h5x/baseimport.py"

app = h5xplorer(context_menu.context_menu,
                baseimport=base,
                extended_selection=False)
