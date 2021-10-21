#!/bin/bash
mkdir -p ./logs/train

# For training based on Sceneflow Data
logdirs=FwSC
cmd="main.py --maxdisp 192
        --model stackhourglass
        --datapath /data2/psmdata/
        --tensorboard_logs=./runs/$logdirs
        --epochs 10
        --savemodel ./checkpoint/$logdirs
        --trainbatchsize 8
        --testbatchsize 4
        --convolution_type FwSC"
echo $cmd
python -W ignore $cmd >> ./logs/train/${logdirs}.txt

logdirs=FDwSC
cmd="main.py --maxdisp 192
        --model stackhourglass
        --datapath /data2/psmdata/
        --tensorboard_logs=./runs/$logdirs
        --epochs 10
        --savemodel ./checkpoint/$logdirs
        --trainbatchsize 8
        --testbatchsize 4
        --convolution_type FDwSC"
echo $cmd
python -W ignore $cmd >> ./logs/train/${logdirs}.txt

# For finetuning based on KITTI2015 Data
logdirs=FwSC
resume=FwSC_PSMNet_sceneflow.tar
cmd="finetune.py --maxdisp 192
            --datatype 2015
            --model stackhourglass
            --datapath /data/rahim/data/Kitti_2015/training/
            --tensorboard_logs ./runs/${logdirs}
            --loadmodel ./checkpoint/FwSC/${resume}
            --epochs 1000
            --savemodel ./checkpoint/FwSC/${logdirs}
            --trainbatchsize 8
            --testbatchsize 4
            --epoch_num 1
            --convolution_type FwSC"
echo $cmd
python -W ignore $cmd >> ./logs/train/${logdirs}.txt


logdirs=FDwSC
resume=FDwSC_PSMNet_sceneflow.tar
cmd="finetune.py --maxdisp 192
            --datatype 2015
            --model stackhourglass
            --datapath /data/rahim/data/Kitti_2015/training/
            --tensorboard_logs ./runs/${logdirs}
            --loadmodel ./checkpoint/FDwSC/${resume}
            --epochs 1000
            --savemodel ./checkpoint/FDwSC/${logdirs}
            --trainbatchsize 8
            --testbatchsize 4
            --epoch_num 1
            --convolution_type FDwSC"
echo $cmd
python -W ignore $cmd >> ./logs/train/${logdirs}.txt