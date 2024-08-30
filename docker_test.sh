# Setup MSTA3D and pointgroup_ops
python setup.py develop
cd msta3d/lib
python setup.py develop
cd ../..

# Inference
python tools/test.py configs/msta3d_scannet.yaml checkpoints/msta3d_scannet.pth