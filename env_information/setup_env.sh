echo "============== Installing Pytorch 1.10.0 =============="
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge

echo "========= Installing some necessary lirabries ========="
pip install spconv-cu113
conda install pytorch-scatter -c pyg
cd MSTA3D/
pip install -r requirements.txt

python setup.py develop
cd msta3d/lib
python setup.py develop
cd ../..
