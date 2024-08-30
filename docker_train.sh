# Setup MSTA3D and pointgroup_ops
python setup.py develop
cd msta3d/lib
python setup.py develop
cd ../..

# Training
python tools/train.py configs/msta3d_scannet.yaml