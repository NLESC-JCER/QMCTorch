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

# log.info("   ____  __  __  _____ _______             _")
# log.info("  / __ \|  \/  |/ ____|__   __|           | |    ")
# log.info(" | |  | | \  / | |       | | ___  _ __ ___| |__  ")
# log.info(" | |  | | |\/| | |       | |/ _ \| '__/ __| '_ \ ")
# log.info(" | |__| | |  | | |____   | | (_) | | | (__| | | |")
# log.info("  \___\_\_|  |_|\_____|  |_|\___/|_|  \___|_| |_|")

log.info("  ____    __  ______________             _")
log.info(" / __ \  /  |/  / ___/_  __/__  ________/ /  ")
log.info("/ /_/ / / /|_/ / /__  / / / _ \/ __/ __/ _ \ ")
log.info("\___\_\/_/  /_/\___/ /_/  \___/_/  \__/_//_/ ")
