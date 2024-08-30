FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get update \
    && apt-get install -y git curl libsparsehash-dev

# Install torch-scatter
RUN pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.0+cu113.html --no-deps


WORKDIR /workspace
COPY . /workspace
# Install Segmentator
RUN apt-get remove -y cmake && \
    curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2-linux-x86_64.sh -o cmake-install.sh && \
    bash cmake-install.sh --skip-license --prefix=/usr/local && \
    rm cmake-install.sh && \
    ln -s /usr/local/bin/cmake /usr/bin/cmake
RUN git clone https://github.com/Karbo123/segmentator.git \
    && cd segmentator/csrc \
    && git reset --hard 76efe46d03dd27afa78df972b17d07f2c6cfb696 \
    && mkdir build \
    && cd build \
    && cmake .. \
        -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
        -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
        -DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'` \
    && make \
    && make install \
    && cd ../../..

# Install Python packages
RUN pip install --no-deps \
    spconv-cu113 \
    gorilla-core \
    numba \
    open3d \
    opencv-python \
    plyfile \
    scipy \
    tensorboard \
    tensorboardX \
    tqdm \
    yapf \
    einops \
    pyviz3d

# Setup MSTA3D & pointgroup_ops
# CMD ["./docker_train.sh"]
CMD ["./docker_test.sh"]

