#!/bin/bash
#SBATCH -n 1
#SBATCH -t 10:00:00

export PYTHONPATH="$PYTHONPATH:/home/dewit/QMCTorch"

cd h2/ || exit
python h2_m.py
python h2_hmc.py

cd ../lih || exit
python lih_m.py
python lih_hmc.py

