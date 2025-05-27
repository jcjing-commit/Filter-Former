#!/bin/bash

#!/bin/bash

# DATA_DIR="../mvtec_ad_2" #Root directory of the MVTec AD 2 dataset
DATA_DIR="./data/mvtec_ad_2" #Root directory of the MVTec AD 2 dataset

# Train call   
python filter_former.py --data_path $DATA_DIR --save_dir "./saved_results" --total_epochs 300 --batch_size 2 --phase 'train'

# test public
python filter_former.py --data_path $DATA_DIR --save_dir "./saved_results"   --work_out "./workout" --batch_size 1 --phase 'test_public'

#geting the Threshold value
python evaluate_tool/evaluate_f1_score.py --anomaly_maps_dir './workout/anomaly_images' --dataset_base_dir $DATA_DIR --output_dir './workout' 

# test test_private
python filter_former.py --data_path $DATA_DIR --save_dir "./saved_results"   --work_out "./workout" --batch_size 1 --phase 'test_private' --thresholded_cfg  './workout/threshold_value.json' 

# test 
python filter_former.py --data_path $DATA_DIR --save_dir "./saved_results"   --work_out "./workout" --batch_size 1 --phase 'test_private_mixed' --thresholded_cfg  './workout/threshold_value.json' 
