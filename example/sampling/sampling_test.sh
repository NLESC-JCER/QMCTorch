#!/bin/bash
#SBATCH -n 1
#SBATCH -t 10:00:00

export PYTHONPATH="$PYTHONPATH:/home/dewit/QMCTorch"
#export PYTHONPATH="$PYTHONPATH:/home/matthijs/esc/QMCTorch"

cd h2/ || exit
python h2_m.py
python h2_hmc.py
python h2_vrs.py

#cd ../lih || exit
#python lih_m.py
#python lih_hmc.py

