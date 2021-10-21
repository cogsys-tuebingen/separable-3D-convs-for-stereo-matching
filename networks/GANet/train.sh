
codedir=.
mkdir -p a ${codedir}/logs/train

# Sceneflow training or resume
logdirs=GANet11_FwSC_sceneflow_train
model=GANet11 #GANet11 or GANet_deep
starting_epoch=1 # can be more than 1 when resuming
resume_point="" # checkpoint to resume from
wandb_project_name=GANet_SF
wandb_run_name=Sceneflow_training
conv_type=FwSC

cmd="train.py --data_path=/data2/rahim/data/
        --crop_height=240
        --crop_width=528
        --model=$model
        --convolution_type=$conv_type
        --save_path=${codedir}/checkpoint/${logdirs}
        --nEpochs=15
        --tensorboard_logs=${codedir}/runs/${logdirs}
        --training_list=${codedir}/lists/sceneflow_train.list
        --val_list=${codedir}/lists/sceneflow_val.list
        --val_save_path=${codedir}/validation-result/${logdirs}
        --resume=${codedir}/checkpoint/${resume_point}
        --epoch_nums=${starting_epoch}
        --wandb_project_name=${wandb_project_name}
        --wandb_run_name=${wandb_run_name}
        --max_disp=192
        --thread=16
        --kitti2015=0
        --shift=3
        --batchSize=4
        --testBatchSize=4"
echo $cmd
which python
python -W ignore $cmd >> ${codedir}/logs/train/${logdirs}.txt
exit

# Finetune Kitti2015

logdirs=GANet11_FwSC_kitti2015_train
model=GANet11
starting_epoch=1
resume_point="FwSC_GANet11_sceneflow" #to finetune from sceneflow checkpoint
wandb_project_name=GANet_kitti
wandb_run_name=kitti_training
conv_type=FwSC
cmd="train.py --data_path=/data/rahim/data/Kitti_2015/training/
        --crop_height=240
        --crop_width=528
        --model=$model
        --convolution_type=${conv_type}
        --save_path=${codedir}/checkpoint/${logdirs}
        --nEpochs=1000
        --tensorboard_logs=${codedir}/runs/${logdirs}
        --training_list=${codedir}/lists/kitti2015_train.list
        --val_list=${codedir}/lists/kitti2015_val.list
        --val_save_path=${codedir}/validation-result/${logdirs}
        --resume=${codedir}/checkpoint/${resume_point}
        --epoch_nums=${starting_epoch}
        --wandb_project_name=${wandb_project_name}
        --wandb_run_name=${wandb_run_name}
        --max_disp=192
        --thread=16
        --kitti2015=1
        --shift=3
        --batchSize=4
        --testBatchSize=4"
echo $cmd
which python
python -W ignore $cmd >> ${codedir}/logs/train/${logdirs}.txt

