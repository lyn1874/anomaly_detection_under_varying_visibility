#!/bin/bash
trap "exit" INT
version=${1?Error: experiment version is not defined}
ckpt_step=${2?Error: ckpt step is not defined}
dt=${3?Error: which dataset am I testing? Avenue_bright or Avenue_rain}
datasetpath=/project_scratch/bo/anomaly_data/

if [ $dt = Avenue_bright ]; then
    python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type frames/testing/ --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0

    for bright in 2 3 4 5 6 7 8 9
    do
        python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type original_$bright --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0
    done
    for bright in 2 4 6 8 10
    do
        python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type heavy_$bright --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0
    done
elif [ $dt == Avenue_rain ]; then
    python3 Testing.py --dataset Avenue_rain --dataset_augment_type raining --dataset_augment_test_type frames/testing/ --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0
    raintype='original heavy torrential'
    for srain in $raintype
    do
        for bright in 2 4 6 8 10
        do
            python3 Testing.py --dataset Avenue_rain --dataset_augment_type raining --dataset_augment_test_type ${srain}_$bright --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0
        done
    done
fi




ckpt_step=40
version=5
data=/project_scratch/bo/anomaly_data/

python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type frames/testing/ --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0

# python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type original_2 --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0

# python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type original_3 --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0

# python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type original_4 --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0

# python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type original_5 --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0

# python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type original_6 --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0

# python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type original_7 --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0

# python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type original_8 --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0

# python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type original_9 --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0

# python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type heavy_2 --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0
# python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type heavy_4 --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0
# python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type heavy_6 --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0
# python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type heavy_8 --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0
# python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type heavy_10 --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0

# # version=2
# # python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type heavy_2 --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0
# # python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type heavy_4 --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0
# # python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type heavy_6 --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0
# # python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type heavy_8 --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0
# # python3 Testing.py --dataset Avenue_bright --dataset_augment_type lightning --dataset_augment_test_type heavy_10 --version $version --ckpt_step $ckpt_step --data_path $data --EntropyLossWeight 0









# # python Train.py --dataset_path /project_scratch/bo/anomaly_data/ --dataset_type UCSDped2 --dataset_augment_type original --exp_dir /project/bo/exp_data/memory_normal/ --version 1 --EntropyLossWeight 0.00005 --lr 1e-4 








