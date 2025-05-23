#!/usr/bin bash

for dataset in 'MIntRec'
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'mag_bert' \
        --method 'mag_bert' \
        --data_mode 'multi-class' \
        --train \
        --save_results \
        --save_model \
        --seed $seed \
        --gpu_id '0' \
        --video_feats_path 'video_feats.pkl' \
        --audio_feats_path 'audio_feats.pkl' \
        --text_backbone 'bert-base-uncased' \
        --config_file_name 'mag_bert' \
        --results_file_name 'mag_bert.csv'
    done
done