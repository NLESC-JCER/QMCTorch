# -*- coding: utf-8 -*-
"""Documentation about QMCTorch"""

from .__version__ import __version__

__author__ = "Nicolas Renaud"
__email__ = 'n.renaud@esciencecenter.nl'

import twiggy
import sys
twiggy.quick_setup(file=sys.stdout)
log = twiggy.log.name('QMCTorch')
log.min_level = twiggy.levels.INFO

# log.info(r"   ____  __  __  _____ _______             _")
# log.info(r"  / __ \|  \/  |/ ____|__   __|           | |    ")
# log.info(r" | |  | | \  / | |       | | ___  _ __ ___| |__  ")
# log.info(r" | |  | | |\/| | |       | |/ _ \| '__/ __| '_ \ ")
# log.info(r" | |__| | |  | | |____   | | (_) | | | (__| | | |")
# log.info(r"  \___\_\_|  |_|\_____|  |_|\___/|_|  \___|_| |_|")

log.info(r"  ____    __  ______________             _")
log.info(r" / __ \  /  |/  / ___/_  __/__  ________/ /  ")
log.info(r"/ /_/ / / /|_/ / /__  / / / _ \/ __/ __/ _ \ ")
log.info(r"\___\_\/_/  /_/\___/ /_/  \___/_/  \__/_//_/ ")
