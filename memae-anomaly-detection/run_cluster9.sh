#!/bin/bash
ds=${1?Error: which type of data am I using? Avenue, Avenue_bright or Avenue_rain}
version=${2?Error: I forget to define the experiment version}

datapath=/project_scratch/bo/anomaly_data/
exppath=/project/bo/exp_data/memory_normal/
if [ -d "$datapath" ]; then
    echo "folder $datapath already exists"
    echo "next step, train the model"
else
    echo "folder $datapath does not exist, please download the data first"
fi

if [ $ds = Avenue_bright ]; then
    python Train.py --dataset_path $datapath --dataset_type Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type lightning --version $version --EntropyLossWeight 0 --lr 1e-4 --exp_dir $exppath --epochs 40  # --batch_size 5

elif [ $ds = Avenue_rain ]; then
    python Train.py --dataset_path $datapath --dataset_type Avenue_rain --dataset_augment_type raining --dataset_augment_test_type raining --version $version --EntropyLossWeight 0 --lr 1e-4 --exp_dir $exppath --epochs 60  # --batch_size 5
fi


