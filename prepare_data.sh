#!/bin/bash
trap "exit" INT
datapath=/project_scratch/bo/anomaly_data/Avenue_play/
expdir=$(pwd)
if [ -d "${datapath}" ]; then
    echo "Avenue dataset exists"
    trainfolder=${datapath}training/
    if [ -d "$trainfolder" ]; then
        echo "Frames already exist, YEAH!"
    else
        echo "Extract frames"
        python3 aug_data.py --option extract --datapath $datapath 
    fi        
else
    echo "Download the Avenue dataset...."
    mkdir $datapath
    cd $datapath
    wget http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip
    unzip Avenue_Dataset.zip
    mv 'Avenue Dataset'/* .
    echo "Successfully download the dataset, next extract frames from the Avenue dataset"
    cd $expdir
    python3 aug_data.py --option extract --datapath $datapath 
fi