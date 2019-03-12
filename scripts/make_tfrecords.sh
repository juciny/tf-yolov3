cat ./video_dataset/labels.txt | head -n  540 > ./video_dataset/train.txt
cat ./video_dataset/labels.txt | tail -n +541 > ./video_dataset/test.txt
python core/convert_tfrecord.py --dataset_txt ./video_dataset/train.txt --tfrecord_path_prefix ./video_dataset/video_train
python core/convert_tfrecord.py --dataset_txt ./video_dataset/test.txt  --tfrecord_path_prefix ./video_dataset/video_test
