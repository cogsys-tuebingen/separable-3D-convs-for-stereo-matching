#export PYTHONPATH=$PWD:$PYTHONPATH
TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")
#echo $TORCH
cd libs/GANet
python setup.py clean
rm -rf build
python setup.py build
cp -r build/lib* build/lib

cd ../sync_bn
python setup.py clean
rm -rf build
python setup.py build
cp -r build/lib* build/lib
