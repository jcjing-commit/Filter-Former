# Train call
python filter_former.py --data_path "../mvtec_ad_2" --save_dir "./saved_results" --total_epochs 1  --batch_size 4 --phase 'train'

# test public
python filter_former.py --data_path "../mvtec_ad_2" --save_dir "./saved_results"   --work_out "./workout" --batch_size 4 --phase 'test_public'

#geting the Threshold value
python evaluate_tool/evaluate_f1_score.py --anomaly_maps_dir './workout/anomaly_images/' --dataset_base_dir '../mvtec_ad_2' --output_dir './workout/' 


# test test_private
python filter_former.py --data_path "../mvtec_ad_2" --save_dir "./saved_results"   --work_out "./workout" --batch_size 4 --phase 'test_private' --thresholded_cfg  './workout/threshold_value.json' 


# test 
python filter_former.py --data_path "../mvtec_ad_2" --save_dir "./saved_results"   --work_out "./workout" --batch_size 4 --phase 'test_private_mixed' --thresholded_cfg  './workout/threshold_value.json' 
 