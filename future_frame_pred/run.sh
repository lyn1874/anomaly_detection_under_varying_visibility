#!/bin/bash
trap "exit" INT
datadir=${1?Error: I forget to define the data path}
version=${2?Error: I forget to define the experiment version}
# python train_own.py --dataset_type Avenue_bright --iters 25000 --save_interval 5000 --val_interval 5000 --dataset_path $datadir --dataset_augment_type lightning --dataset_augment_test_type lightning --version $version

python train_own.py --dataset_type Avenue_bright --iters 75000 --save_interval 5000 --val_interval 5000 --dataset_path $datadir --dataset_augment_type lightning --dataset_augment_test_type lightning --version $version

