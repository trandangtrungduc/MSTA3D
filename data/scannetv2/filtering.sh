#!/bin/bash
echo Find noisy superpoint
python find_noisy_spp.py --subset train --sim_threshold 0.83
python find_noisy_spp.py --subset val --sim_threshold 0.83
echo Create a new dataset
python create_new_dataset.py --subset train --point_threshold 300
python create_new_dataset.py --subset val --point_threshold 300
