codedir=.
mkdir -p ./logs/predict
# KITTI2015 DATASET
logdirs=FwSC_GANet11_kitti2015
resume=FwSC_GANet11_sceneflow_finetuned_kitti15.pth #see evaluate.sh for other checkpoints names

cmd="predict.py --crop_height=384
                  --crop_width=1248
                  --max_disp=192
                  --model=GANet11
                  --data_path=/data/rahim/data/Kitti_2015/testing/
                  --test_list=${codedir}/lists/kitti2015_test.list
                  --save_path=${codedir}/predict-result/${logdirs}/
                  --resume=${codedir}/checkpoint/FwSC/${resume}
                  --kitti2015=1
                  --kitti=0
                  --convolution_type=FwSC "
echo $cmd
python -W ignore $cmd >> ${codedir}/logs/predict/${logdirs}.txt


# KITTI2012 DATASET
#logdirs=predict_kitti2012
#resume="kitti2012_final.pth"
#cmd="predict.py --crop_height=384
#                  --crop_width=1248
#                  --max_disp=192
#                  --data_path=/data/rahim/data/Kitti_2012/testing/
#                  --test_list=${codedir}/lists/kitti2012_test.list
#                  --save_path=${codedir}/predict-result/${logdirs}/
#                  --resume=${codedir}/checkpoint/${resume}
#                  --kitti=1
#                  --convolution_type=FwSC"
#echo $cmd
#which python
#python -W ignore $cmd  >> ${codedir}/logs/predict/${logdirs}.txt
#exit