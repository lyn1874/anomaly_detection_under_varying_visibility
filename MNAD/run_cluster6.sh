#!/bin/bash
ds=${1?Error: which dataset am I using? Avenue, Avenue_bright, or Avenue_rain}
datadir=${2?Error: what is the path for the data? /project_scratch/bo/anomaly_data/}
dat=${3?Error: augmenting method, training, lightning or raining}
expdir=/project/bo/exp_data/MNAD_exp/
ver=0

if [ $ds = Avenue ]; then
    python Train.py --dataset_path $datadir --dataset_type $ds --dataset_augment_type training --dataset_augment_test_type testing --exp_dir $expdir --version $ver --decay_step 0 --epochs 40 --lr 2e-4 --batch_size 10
    
elif [ $ds = Avenue_bright ]; then
    python Train.py --dataset_path $datadir --dataset_type $ds --dataset_augment_type lightning --dataset_augment_test_type lightning --exp_dir $expdir --version $ver --decay_step 0 --epochs 40 --lr 2e-4 --batch_size 10
    
elif [ $ds = Avenue_rain ]; then
    python Train.py --dataset_path $datadir --dataset_type $ds --dataset_augment_type raining --dataset_augment_test_type raining --exp_dir $expdir --version $ver --decay_step 0 --epochs 40 --lr 2e-4 --batch_size 10

fi

    

