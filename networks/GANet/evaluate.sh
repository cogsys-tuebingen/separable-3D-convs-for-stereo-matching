# For MiddBurry Data set
#--crop_height=432
#--crop_width=672
#--max_disp=192
#--data_path=/data/rahim/data/MiddEval3/
#--test_list=lists/middeval.list
#--save_path=./result/
#--mideval=1
#--kitti=0
#--kitti2015=0
#--resume=./models/sceneflow_epoch_10.pth

# For kitti2012 Data set
#--crop_height=384
#--crop_width=1248
#--max_disp=48
#--data_path=/data/rahim/data/Kitti_2012/training/
#--test_list=lists/kitti2012_val24.list
#--save_path=./result/
#--kitti2015=0
#--kitti=1
#--resume=./models/kitti2012_final.pth

mkdir -p ./logs/evaluate

# KITTI2015 DATASET
logdirs=FwSC_GANet11_kitti2015
resume=FwSC_GANet11_sceneflow_finetuned_kitti15.pth
cmd="evaluate.py --crop_height=384
            --crop_width=1248
            --max_disp=192
            --data_path=/data/rahim/data/Kitti_2015/training/
            --test_list=lists/kitti2015_val.list
            --save_path=./evaluation-results/${logdirs}
            --kitti2015=1
            --kitti=0
            --resume=./checkpoint/FwSC/$resume
            --model=GANet11
            --max_test_images=10
            --convolution_type=FwSC"
echo $cmd
python -W ignore $cmd >> ./logs/evaluate/${logdirs}.txt


logdirs=FwSC_GANetdeep_kitti2015
resume=FwSC_GANetdeep_sceneflow_finetuned_kitti15.pth
cmd="evaluate.py --crop_height=384
            --crop_width=1248
            --max_disp=192
            --data_path=/data/rahim/data/Kitti_2015/training/
            --test_list=lists/kitti2015_val.list
            --save_path=./evaluation-results/${logdirs}
            --kitti2015=1
            --kitti=0
            --resume=./checkpoint/FwSC/$resume
            --model=GANet_deep
            --max_test_images=10
            --convolution_type=FwSC"
echo $cmd
python -W ignore $cmd >> ./logs/evaluate/${logdirs}.txt


logdirs=FDwSC_GANet11_kitti2015
resume=FDwSC_GANet11_sceneflow_finetuned_kitti15.pth

cmd="evaluate.py --crop_height=384
            --crop_width=1248
            --max_disp=192
            --data_path=/data/rahim/data/Kitti_2015/training/
            --test_list=lists/kitti2015_val.list
            --save_path=./evaluation-results/${logdirs}
            --kitti2015=1
            --kitti=0
            --resume=./checkpoint/FDwSC/$resume
            --model=GANet11
            --max_test_images=10
            --convolution_type=FDwSC"
echo $cmd
python -W ignore $cmd >> ./logs/evaluate/${logdirs}.txt


logdirs=FDwSC_GANetdeep_kitti2015
resume=FDwSC_GANetdeep_sceneflow_finetuned_kitti15.pth

cmd="evaluate.py --crop_height=384
            --crop_width=1248
            --max_disp=192
            --data_path=/data/rahim/data/Kitti_2015/training/
            --test_list=lists/kitti2015_val.list
            --save_path=./evaluation-results/${logdirs}
            --kitti2015=1
            --kitti=0
            --resume=./checkpoint/FDwSC/$resume
            --model=GANet_deep
            --max_test_images=10
            --convolution_type=FDwSC"
echo $cmd
python -W ignore $cmd >> ./logs/evaluate/${logdirs}.txt


# SCENEFLOW DATASET
logdirs=FwSC_GANet11_sceneflow
resume=FwSC_GANet11_sceneflow.pth
cmd="evaluate.py --crop_height=576
            --crop_width=960
            --max_disp=192
            --model=GANet11
            --data_path=/data2/rahim/data/
            --test_list=./lists/sceneflow_val.list
            --save_path=./evaluation-results/${logdirs}
            --kitti2015=0
            --kitti=0
            --mideval=0
            --print_GT=0
            --resume=./checkpoint/FwSC/$resume
            --max_test_images=10
            --print_input_images=0
            --convolution_type=FwSC"
echo $cmd
python -W ignore $cmd >> ./logs/evaluate/${logdirs}.txt


logdirs=FwSC_GANetdeep_sceneflow
resume=FwSC_GANetdeep_sceneflow.pth
cmd="evaluate.py --crop_height=576
            --crop_width=960
            --max_disp=192
            --model=GANet_deep
            --data_path=/data2/rahim/data/
            --test_list=./lists/sceneflow_val.list
            --save_path=./evaluation-results/${logdirs}
            --kitti2015=0
            --kitti=0
            --mideval=0
            --print_GT=0
            --resume=./checkpoint/FwSC/$resume
            --max_test_images=10
            --print_input_images=0
            --convolution_type=FwSC"
echo $cmd
python -W ignore $cmd >> ./logs/evaluate/${logdirs}.txt


logdirs=FDwSC_GANet11_sceneflow
resume=FDwSC_GANet11_sceneflow.pth
cmd="evaluate.py --crop_height=576
            --crop_width=960
            --max_disp=192
            --model=GANet11
            --data_path=/data2/rahim/data/
            --test_list=./lists/sceneflow_val.list
            --save_path=./evaluation-results/${logdirs}
            --kitti2015=0
            --kitti=0
            --mideval=0
            --print_GT=0
            --resume=./checkpoint/FDwSC/$resume
            --max_test_images=10
            --print_input_images=0
            --convolution_type=FDwSC"
echo $cmd
python -W ignore $cmd >> ./logs/evaluate/${logdirs}.txt

logdirs=FDwSC_GANetdeep_sceneflow
resume=FDwSC_GANetdeep_sceneflow.pth
cmd="evaluate.py --crop_height=576
            --crop_width=960
            --max_disp=192
            --model=GANet_deep
            --data_path=/data2/rahim/data/
            --test_list=./lists/sceneflow_val.list
            --save_path=./evaluation-results/${logdirs}
            --kitti2015=0
            --kitti=0
            --mideval=0
            --print_GT=0
            --resume=./checkpoint/FDwSC/$resume
            --max_test_images=10
            --print_input_images=0
            --convolution_type=FDwSC"
echo $cmd
python -W ignore $cmd >> ./logs/evaluate/${logdirs}.txt
exit