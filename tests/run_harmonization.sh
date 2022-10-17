#!/bin/bash

export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# activate the virtual environment
source "/home/nnieto/Nico/Harmonization/Harmonization/bin/activate"

cd juharmonize-main
# run your script
python3 test_all_data_parallel.py --data_dir /home/nnieto/Nico/Harmonization/data/ --save_dir /home/nnieto/Nico/Harmonization/data/ --n_high_var_feats 10 --sites_use HCP IXI --sites_oos eNKI --problem_type binary_classification --harmonize_mode Cheat