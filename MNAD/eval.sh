#!/bin/bash
trap "exit" INT
ds=${1?Error: Dataset type, Avenue, Avenue_bright or Avenue_rain}
version=${2?Error: experiment version is not defined}
ckpt_step=${3?Error: ckpt step is not defined}

datasetpath=/project_scratch/bo/anomaly_data/
expdir=/project/bo/exp_data/MNAD_exp/

if [ $ds = Avenue_bright ]; then
    python3 Test.py --dataset_type Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type frames/testing/ --exp_dir $expdir --version $version --decay_step 0 --ckpt_step $ckpt_step --dataset_path $datasetpath
    for bright in 2 3 4 5 6 7 8 9
    do
        python3 Test.py --dataset_type Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type original_$bright --exp_dir $expdir --version $version --decay_step 0 --ckpt_step $ckpt_step --dataset_path $datasetpath
    done
    for bright in 2 4 6 8 10
    do
        python3 Test.py --dataset_type Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type heavy_$bright --exp_dir $expdir --version $version --decay_step 0 --ckpt_step $ckpt_step --dataset_path $datasetpath
    done
elif [ $ds = Avenue_rain ]; then
    raintype='original heavy torrential'
    for s in $raintype
    do
        for bright in 2 4 6 8 10
        do
            python3 Test.py --dataset_type Avenue_rain --dataset_augment_type raining --dataset_augment_test_type ${s}_$bright --exp_dir $expdir --version $version --decay_step 0 --ckpt_step $ckpt_step --dataset_path $datasetpath
        done
    done
fi


