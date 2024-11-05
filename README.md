# [MSTA3D: Multi-scale Twin-Attention for 3D Instance Segmentation](https://arxiv.org/abs/2411.01781)

Duc Dang Trung Tran, Byeongkeun Kang, Yeejin Lee

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/msta3d-multi-scale-twin-attention-for-3d/3d-instance-segmentation-on-scannetv2)](https://paperswithcode.com/sota/3d-instance-segmentation-on-scannetv2?p=msta3d-multi-scale-twin-attention-for-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/msta3d-multi-scale-twin-attention-for-3d/3d-object-detection-on-scannetv2)](https://paperswithcode.com/sota/3d-object-detection-on-scannetv2?p=msta3d-multi-scale-twin-attention-for-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/msta3d-multi-scale-twin-attention-for-3d/3d-instance-segmentation-on-scannet200)](https://paperswithcode.com/sota/3d-instance-segmentation-on-scannet200?p=msta3d-multi-scale-twin-attention-for-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/msta3d-multi-scale-twin-attention-for-3d/3d-instance-segmentation-on-s3dis)](https://paperswithcode.com/sota/3d-instance-segmentation-on-s3dis?p=msta3d-multi-scale-twin-attention-for-3d)

## :electron: Overall Architecture :electron

<img src="docs\overall_structure.png" />

## :tada: Introduction :tada:

Most existing methods suffer from two drawbacks:
* Over-segmentation issue on large instances and background noise.
* The produced superpoint mask is not really reliable when we convert it to point cloud mask.

Solution:
* Using multi-scale superpoint combined with a twin-attention decoder to capture objects of different size and shape.
* Using spatial constraint regularizer helps the model create a more reliable superpoint mask.

## :link: Citation :link:

[Paper (arXiv)](https://arxiv.org/abs/2411.01781) is accepted at ACM Multimedia 2024.
```
@inproceedings{tran2024msta3d,
  title={MSTA3D: Multi-scale Twin-attention for 3D Instance Segmentation},
  author={Tran, Duc Dang Trung and Kang, Byeongkeun and Lee, Yeejin},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={1467--1475},
  year={2024}
}
```
## :hammer_and_wrench: Installation :hammer_and_wrench:

Requirements

The following installation suppose `python=3.8` `pytorch=1.10.0` and `cuda=11.3`

- Clone the repository

  ```
  git clone https://github.com/trandangtrungduc/MSTA3D.git
  cd MSTA3D
  ```

- Environment Installation
  ```
  ./setup_env.sh
  ```
- Refer to `env_information/msta3d_environment.yml` for details of the environment we used.

## :nut_and_bolt: Data Preprocessing :nut_and_bolt:

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

## :arrow_down: Pretrained Model :arrow_down:

Download [SSTNet](https://drive.google.com/file/d/1vucwdbm6pHRGlUZAYFdK9JmnPVerjNuD/view?usp=sharing) pretrained model


```
mkdir checkpoints
mv ${SSTNET_PRETRAINED_PATH}/sstnet_pretrain.pth checkpoints/
```

## :airplane: Training :airplane:
ScanNetV2 dataset

```
python tools/train.py configs/msta3d_scannet.yaml
```
ScanNet200 dataset

```
python tools/train.py configs/msta3d_scannet200.yaml
```

## :straight_ruler: Testing :straight_ruler:
ScannetV2 dataset
```
python tools/test.py configs/msta3d_scannet.yaml ${CHECKPOINT_PATH}/msta3d_scannet.pth
```
ScanNet200 dataset
```
python tools/test.py configs/msta3d_scannet200.yaml ${CHECKPOINT_PATH}/ msta3d_scannet200.pth
```

## :checkered_flag: Checkpoints :checkered_flag:
The results in the table may change a little because of randomness.
Put the downloaded checkpoint files to `checkpoints/`

| Dataset | mAP | mAP<sub>50</sub> | mAP<sub>25</sub>  | Download |
|:-------:|:----------------:|:----------------:|:----:|:--------:|
| ScanNetV2 | 58.4 | 77.0 | 85.4 | [model](https://huggingface.co/TDTDuc/MSTA3D/tree/main) &#124; [config](configs/msta3d_scannet.yaml) |
| ScanNet200 | 26.2 | 35.2 | 40.1 | [model](https://huggingface.co/TDTDuc/MSTA3D/tree/main) &#124; [config](configs/msta3d_scannet200.yaml) |


## :art: Visualization :art:

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

## :rocket: Examples :rocket:

<img src="docs\example.png" />

## :gift: Acknowledgement :gift:

Sincerely thanks for [SPFormer](https://github.com/sunjiahao1999/SPFormer/tree/main) repository. This repository is build upon them.
