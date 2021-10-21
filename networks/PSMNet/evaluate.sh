mkdir -p ./logs/evaluate

# KITTI2015 DATASET
logdirs=FwSC_PSMNet_kitti2015
resume=FwSC_PSMNet_sceneflow_finetuned_kitti15.tar
cmd="evaluation.py --max_disp 192
                  --model stackhourglass
                  --datapath /data/rahim/data/Kitti_2015/training/
                  --loadmodel ./checkpoint/FwSC/$resume
                  --save_path ./evaluation-results/$logdirs
                  --print_GT 0
                  --print_input_images 0
                  --max_test_images 10
                  --test_list ./lists/kitti2015_val.list
                  --kitti2015 1
                  --convolution_type=FwSC"
echo $cmd
python -W ignore $cmd >> ./logs/evaluate/${logdirs}.txt

logdirs=FDwSC_PSMNet_kitti2015
resume=FDwSC_PSMNet_sceneflow_finetuned_kitti15.tar
cmd="evaluation.py --max_disp 192
                  --model stackhourglass
                  --datapath /data/rahim/data/Kitti_2015/training/
                  --loadmodel ./checkpoint/FDwSC/$resume
                  --save_path ./evaluation-results/$logdirs
                  --print_GT 0
                  --print_input_images 0
                  --max_test_images 10
                  --test_list ./lists/kitti2015_val.list
                  --kitti2015 1
                  --convolution_type=FDwSC"
echo $cmd
python -W ignore $cmd >> ./logs/evaluate/${logdirs}.txt

# SCENEFLOW DATASET
logdirs=FwSC_PSMNet_sceneflow
resume=FwSC_PSMNet_sceneflow.tar
cmd="evaluation.py --max_disp 192
                --model stackhourglass
                --datapath /data2/psmdata/
                --loadmodel ./checkpoint/FwSC/$resume
                --save_path ./evaluation-results/$logdirs
                --print_GT 0
                --print_input_images 10
                --max_test_images 10
                --test_list ./lists/sceneflow_test.list
                --kitti2015 0
                --convolution_type=FwSC"
echo $cmd
python -W ignore $cmd >> ./logs/evaluate/${logdirs}.txt



logdirs=FDwSC_PSMNet_sceneflow
resume=FDwSC_PSMNet_sceneflow.tar
cmd="evaluation.py --max_disp 192
                --model stackhourglass
                --datapath /data2/psmdata/
                --loadmodel ./checkpoint/FDwSC/$resume
                --save_path ./evaluation-results/$logdirs
                --print_GT 0
                --print_input_images 10
                --max_test_images 10
                --test_list ./lists/sceneflow_test.list
                --kitti2015 0
                --convolution_type=FDwSC"
echo $cmd
python -W ignore $cmd >> ./logs/evaluate/${logdirs}.txt
exit