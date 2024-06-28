#!/bin/bash

sudo apt-get install libsparsehash-dev

# Local variables
PROJECT=msta3d
PROJECT_DIR=${PWD}
PYTHON=3.8
TORCH=1.10.0
CUDA_SUPPORTED=(11.3 11.4)
CUDA_VERSION=`nvcc --version | grep release | sed 's/.* release //' | sed 's/, .*//'`

# Project installation
echo " 🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥"
echo "🐥                                          🐥"
echo "🐥    🌀  Multi-scale Twin-attention  🌀    🐥"
echo "🐥    🌀 for 3D Instance Segmentation 🌀    🐥"
echo "🐥    🌀   Environment Installation   🌀    🐥"
echo "🐥                                          🐥"
echo " 🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥🐥"
echo
echo "****************************"
echo
echo "Project       : ${PROJECT}"
echo "Python version: ${PYTHON}"
echo "Torch version : ${TORCH}"
echo "Current CUDA  : ${CUDA_VERSION}"
echo
echo "****************************"

echo
echo "🔎 Checking for installed CUDA 🔍"
echo

# CUDA checking
CUDA_MAJOR=`echo ${CUDA_VERSION} | sed 's/\..*//'`
CUDA_MINOR=`echo ${CUDA_VERSION} | sed 's/.*\.//'`

if [[ ! " ${CUDA_SUPPORTED[*]} " =~ " ${CUDA_VERSION} " ]]
then
    echo "CUDA ${CUDA_VERSION} is not among the supported versions: "`echo ${CUDA_SUPPORTED[*]}`
    echo "Please install CUDA to one of the supported versions."
    exit 1
else
    echo "Found CUDA ${CUDA_VERSION} installed among the supported versions."
fi

echo
echo "🔎 Checking for installed Anaconda 🔍"
echo

# Conda checking
CONDA_DIR=`realpath ~/miniconda3`
if (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
then
  echo "Found anaconda at ${CONDA_DIR}."
  CONDA_DIR=`realpath ~/anaconda3`
fi

while (test -z $CONDA_DIR) || [ ! -d $CONDA_DIR ]
do
    echo "Could not find conda at: "$CONDA_DIR
    read -p "Please provide your conda install directory: " CONDA_DIR
    CONDA_DIR=`realpath $CONDA_DIR`
done

echo "Using conda at: ${CONDA_DIR}/etc/profile.d/conda.sh"
source ${CONDA_DIR}/etc/profile.d/conda.sh

echo
echo " 🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍"
echo "🐍                                        🐍"
echo "🐍       Creating Conda Environment       🐍"
echo "🐍                                        🐍"
echo " 🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍🐍"
echo

conda create --name ${PROJECT} python=${PYTHON} -y
source ${CONDA_DIR}/etc/profile.d/conda.sh
conda activate ${PROJECT}

echo "🎁🎁🎁 Successfully created conda environment 🎁🎁🎁🎁"

echo
echo " 🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨"
echo "🔨                                        🔨"
echo "🔨       Installing Dependencies          🔨"
echo "🔨                                        🔨"
echo " 🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨🔨"
echo

pip3 install torch==${TORCH} torchvision --index-url https://download.pytorch.org/whl/cu${CUDA_MAJOR}${CUDA_MINOR}
echo "🎁🎁🎁 Successfully installed torch ${TORCH} with CUDA ${CUDA_VERSION}🎁🎁🎁"

pip install torch-scatter==2.0.9
pip install spconv-cu113==2.3.6
pip install gorilla-core==0.2.7.8
pip install numba==0.58.1
pip install open3d==0.18.0
pip install opencv-python==4.9.0.80
pip install plyfile==1.0.3
pip install scipy==1.10.1
pip install tensorboard==2.14.0
pip install tensorboardX==2.6.2.2
pip install tqdm==4.66.2
pip install yapf==0.40.2
pip install einops==0.7.0
pip install pyviz3d==0.3.0

echo "🎁🎁🎁 Successfully installed dependencies 🎁🎁🎁"

echo
echo "🔩 Installing Segmentator for preprocessing 🔩"
echo

cd ${PROJECT_DIR}
git clone https://github.com/Karbo123/segmentator.git
cd segmentator
cd csrc && mkdir build && cd build

cmake .. \
-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
-DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'`

make && make install

echo "🎁🎁🎁 Successfully installed segmentator 🎁🎁🎁"

echo
echo " 💻💻💻💻💻💻💻💻💻💻💻💻💻💻💻💻💻💻💻💻💻"
echo "💻                                        💻"
echo "💻     Installing MSTA3D & Pointgroup     💻"
echo "💻                                        💻"
echo " 💻💻💻💻💻💻💻💻💻💻💻💻💻💻💻💻💻💻💻💻💻"
echo

cd ../../..
python setup.py develop
cd msta3d/lib/
python setup.py develop
cd ../..

echo "🎁🎁🎁 Successfully installed MSTA3D & pointgroup 🎁🎁🎁"

echo
echo " 🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊"
echo "🎊                                    🎊"
echo "🎊   Successfully installed MSTA3D    🎊"
echo "🎊                                    🎊"
echo " 🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊🎊"
