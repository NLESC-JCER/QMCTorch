#!/bin/bash

export PYTHONPATH="$PYTHONPATH:/home/matthijs/esc/QMCTorch/"

cd h2/ || exit
#python h2_m.py
#python h2_hmc.py

cd ../lih || exit
#python lih_m.py
#python lih_hmc.py

cd ../li2 || exit
#python li2_m.py
python li2_hmc.py

cd ../ch4 || exit
#python ch4_m.py
#python ch4_hmc.py
