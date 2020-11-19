#!/bin/bash
#./requirement.shtrap "exit" INT
trap "exit" INT
version=${1?Error: experiment version is not defined}
ckpt_step=${2?Error: ckpt step is not defined}
dt=${3?Error: which dataset am I testing? Avenue_bright or Avenue_rain}
datasetpath=/project_scratch/bo/anomaly_data/

if [ $dt = Avenue_bright ]; then
    python3 evaluate_own.py --dataset_type Avenue_bright --dataset_augment_test_type frames/testing/ --version $version --ckpt_step $ckpt_step --dataset_path $datasetpath
    for bright in 2 3 4 5 6 7 8 9
    do
        python3 evaluate_own.py --dataset_type Avenue_bright --dataset_augment_test_type original_$bright --version $version --ckpt_step $ckpt_step --dataset_path $datasetpath
    done
    for bright in 2 4 6 8 10
    do
        python3 evaluate_own.py --dataset_type Avenue_bright --dataset_augment_test_type heavy_$bright --version $version --ckpt_step $ckpt_step --dataset_path $datasetpath
    done
elif [ $dt == Avenue_rain ]; then
    python3 evaluate_own.py --dataset_type $dt --dataset_augment_test_type frames/testing/ --version $version --ckpt_step $ckpt_step --dataset_path $datasetpath
    raintype='original heavy torrential'
    for srain in $raintype
    do
        for bright in 2 4 6 8 10
        do
            python3 evaluate_own.py --dataset_type $dt --dataset_augment_test_type ${srain}_$bright --version $version --ckpt_step $ckpt_step --dataset_path $datasetpath
        done
    done
fi


