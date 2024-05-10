# MSTA3D: Multi-scale Twin-Attention for 3D Instance Segmentation

[MSTA3D: Multi-scale Twin-Attention for 3D Instance Segmentation]()

Tran Dang Trung Duc, Byeongkeun Kang, Yeejin Lee

## Overall Architecture

<img src="docs\overall_structure.png" />

## Introduction

Most existing methods suffer from two drawbacks:
* Over-segmentation issue on large instances and background noise.
* The produced superpoint mask is not really reliable when we convert it to point cloud mask.

Solution:
* Using multi-scale superpoint combined with a twin-attention decoder to capture objects of different size and shape.
* Using spatial constraint regularizer helps the model create a more reliable superpoint mask.

<img src="docs\benchmark_snapshot.png" alt="snapshot" style="zoom:50%;" />

The snapshot from ScanNetV2 benchmark testing server on 11/04/2024.

## Installation

Requirements

The following installation suppose `python=3.8` `pytorch=1.10.0` and `cuda=11.3`

- Clone the repository

  ```
  git clone https://github.com/trandangtrungduc/MSTA3D.git
  cd MSTA3D
  ```

- Create a conda environment

  ```
  conda create -n msta3d python=3.8
  conda activate msta3d
  ```

- Install the dependencies

  Install [Pytorch 1.10](https://pytorch.org/get-started/previous-versions/)

  ```
  conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
  pip install spconv-cu113
  conda install pytorch-scatter -c pyg
  pip install -r requirements.txt
  ```

  Install segmentator from this [repository](https://github.com/Karbo123/segmentator).

- Setup msta3d and pointgroup_ops

  ```
  sudo apt-get install libsparsehash-dev
  python setup.py develop
  cd msta3d/lib/
  python setup.py develop
  cd ../..
  ```
- Refer to `msta3d_environment.yml` for details of the environment we used.

## Data Preprocessing

### ScanNetV2 dataset

Download the [ScanNet](http://www.scan-net.org/) dataset.

Put the downloaded `scans` and `scans_test` folder as follows

```
MSTA3D
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
```

Split and preprocess data

```
cd data/scannetv2
bash prepare_data.sh
```

After running the script, the scannetv2 dataset structure should look like below

```
MSTA3D
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── val_gt
```
### ScanNet200 dataset

Preprocess data

```
cd data/scannet200
bash prepare_data.sh
```

After running the script the scannet200 dataset structure should look like below

```
MSTA3D
├── data
│   ├── scannet200
│   │   ├── train
│   │   ├── val
```

## Pretrained Model

Download [SSTNet](https://drive.google.com/file/d/1vucwdbm6pHRGlUZAYFdK9JmnPVerjNuD/view?usp=sharing) pretrained model


```
mkdir checkpoints
mv ${SSTNET_PRETRAINED_PATH}/sstnet_pretrain.pth checkpoints/
```

## Training
ScanNetV2 dataset

```
python tools/train.py configs/msta3d_scannet.yaml
```
ScanNet200 dataset

```
python tools/train.py configs/msta3d_scannet200.yaml
```

## Testing
ScannetV2 dataset
```
python tools/test.py configs/msta3d_scannet.yaml ${CHECKPOINT_PATH}/msta3d_scannet.pth
```
ScanNet200 dataset
```
python tools/test.py configs/msta3d_scannet200.yaml ${CHECKPOINT_PATH}/ msta3d_scannet200.pth
```

## Checkpoints
The results in the table may change a little because of randomness

| Dataset | mAP | mAP<sub>50</sub> | mAP<sub>25</sub>  | Download |
|:-------:|:----------------:|:----------------:|:----:|:--------:|
| ScanNetV2 | 58.4 | 77.0 | 85.4 | [model]() &#124; [log](exps/scannetv2/msta3d_scannet/msta3d_scannet.log) &#124; [config](exps/scannetv2/msta3d_scannet/msta3d_scannet.yaml) |
| ScanNet200 | 26.2 | 35.2 | 40.1 | [model]() &#124; [log](exps/scannet200/msta3d_scannet200/msta3d_scannet200.log) &#124;[config](exps/scannet200/msta3d_scannet200/msta3d_scannet200.yaml) |


## Visualization :art:

Before visualization, write the output results of inference

```
mkdir output/
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out output/
```

After writing the results, run visualization by execute the following command

```
python tools/visualize.py --prediction_path output/ --room_name ${SCENE_NAME}
```

You can visualize by Open3D or MeshLab with `.ply` file.

## Examples

<img src="docs\Example.png" />

## Citation

If you find this work useful in your research, please cite:

```
@misc{,
Url = {},
Author = {Duc Tran Dang Trung, Byeongkeun Kang, Yeejin Lee},
Title = {MSTA3D: Multi-scale Twin-Attention for 3D Instance Segmentation},
Publisher = {arXiv},
Year = {2024},
}
```

## Acknowledgement

Sincerely thanks for [SPFormer](https://github.com/sunjiahao1999/SPFormer/tree/main) repository. This repository is build upon them.
