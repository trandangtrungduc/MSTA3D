echo Preprocess train_val data
python3 preprocess_scannet200.py --dataset_root ../scannetv2/scans --output_root ./ --label_map_file ../scannetv2/scannetv2-labels.combined.tsv --train_val_splits_path ../scannetv2/
echo Soft link data
ln -s ../scannetv2/test ./
echo Preprocess Superpoints
python3 preprocess_superpoint.py --dataset_root ../scannetv2 --output_root ./ --data_split train
python3 preprocess_superpoint.py --dataset_root ../scannetv2 --output_root ./ --data_split val