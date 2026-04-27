# BEV Dynamic Tracking

<div align="center">

**面向复杂场景的 BEV 三维单目标跟踪研究**  
**BEV-based 3D Single Object Tracking for Complex Driving Scenarios**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x%2F2.x-orange)]()
[![Task](https://img.shields.io/badge/Task-3D%20Single%20Object%20Tracking-brightgreen)]()
[![Dataset](https://img.shields.io/badge/Dataset-KITTI%20%7C%20nuScenes%20%7C%20Waymo-lightgrey)]()
[![License](https://img.shields.io/badge/License-MIT-blue)]()

</div>

---

## 1. Introduction

This repository contains the official implementation of **BEV Dynamic Tracking**, a BEV-based 3D single object tracking framework for autonomous driving and intelligent transportation scenarios.

3D single object tracking aims to continuously estimate the state of a target object from sequential LiDAR point clouds. Although BEV representation provides a strong trade-off between efficiency and spatial expressiveness, practical tracking systems still suffer from several challenges:

- **Sparse and incomplete LiDAR observations**, especially for pedestrians and cyclists.
- **Frequent occlusion and background interference** in complex traffic scenes.
- **Weak temporal interaction** between historical frames and the current search region.
- **Tracking drift** when the target undergoes abrupt motion or partial observation degradation.

To address these issues, we build a complete tracking pipeline based on BEV representation and introduce two cooperative enhancement modules:

- **TIW: Temporal Interaction Weighting**  
  Strengthens historical-current frame interaction and adaptively aggregates effective temporal cues.

- **DWM: Dynamic Weight Modulation**  
  Dynamically recalibrates feature responses to emphasize target-related regions and suppress redundant noise.

The overall goal is to achieve **stable, accurate, and efficient 3D tracking** under complex driving conditions.

---

## 2. Key Features

- **BEV-based efficient tracking pipeline**  
  Converts raw LiDAR point clouds into compact BEV representations for efficient feature extraction and state prediction.

- **Temporal interaction enhancement**  
  Models the relationship between historical and current frames to improve tracking continuity under occlusion and sparsity.

- **Dynamic feature modulation**  
  Adaptively adjusts feature importance according to scene changes, improving target response and background suppression.

- **End-to-end training and evaluation**  
  Supports training, testing, visualization, and metric evaluation within a unified engineering pipeline.

- **Multi-dataset validation**  
  Designed for experiments on **KITTI**, with extensible support for **nuScenes** and **Waymo**.

- **Competition and research ready**  
  Suitable for academic research, innovation competitions, and engineering demonstration of autonomous driving perception systems.

---

## 3. Framework Overview

The proposed system follows an **Input → BEV Representation → Feature Enhancement → State Prediction → Evaluation** pipeline.

```text
Raw LiDAR Sequence
        │
        ▼
Point Cloud Pre-processing
        │
        ▼
BEV Mapping / ROI Search
        │
        ▼
BEV Backbone Feature Extraction
        │
        ├── Historical BEV Features
        └── Current Search BEV Features
        │
        ▼
TIW: Temporal Interaction Weighting
        │
        ▼
DWM: Dynamic Weight Modulation
        │
        ▼
Tracking Head
        │
        ▼
3D Box Prediction + Tracking Result Visualization
```

### 3.1 BEV Representation

The input point cloud sequence is first processed into BEV feature maps. Compared with point-based representations, BEV features are more deployment-friendly and can better support efficient convolutional or transformer-based feature extraction.

### 3.2 TIW: Temporal Interaction Weighting

TIW is designed to enhance temporal continuity. It aligns and interacts historical features with current search features, then generates temporal weights to aggregate useful historical information.

Main functions:

1. Map historical and current features into a unified representation space.
2. Estimate temporal interaction weights between frames.
3. Aggregate effective historical cues with residual fusion.
4. Improve robustness under occlusion, sparsity, and partial target degradation.

### 3.3 DWM: Dynamic Weight Modulation

DWM focuses on adaptive feature response calibration. In complex scenes, not all BEV regions contribute equally to tracking. DWM dynamically adjusts feature importance to highlight foreground target regions and suppress background interference.

Main functions:

1. Generate dynamic feature weights according to the current scene.
2. Strengthen target-related responses.
3. Reduce redundant or noisy background activations.
4. Improve localization stability and tracking precision.

---

## 4. Repository Structure

A recommended repository structure is shown below. Please adjust the paths according to your actual implementation.

```text
BEV-Dynamic-Tracking/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── configs/
│   ├── kitti/
│   │   ├── car.yaml
│   │   ├── pedestrian.yaml
│   │   ├── van.yaml
│   │   └── cyclist.yaml
│   ├── nuscenes/
│   └── waymo/
│
├── data/
│   ├── kitti/
│   ├── nuscenes/
│   └── waymo/
│
├── docs/
│   ├── figures/
│   └── assets/
│
├── tools/
│   ├── train.py
│   ├── test.py
│   ├── eval.py
│   ├── visualize.py
│   └── benchmark.py
│
├── tracker/
│   ├── datasets/
│   ├── models/
│   │   ├── backbone/
│   │   ├── bev/
│   │   ├── modules/
│   │   │   ├── tiw.py
│   │   │   └── dwm.py
│   │   └── heads/
│   ├── losses/
│   ├── utils/
│   └── visualization/
│
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── results/
│
└── scripts/
    ├── prepare_kitti.sh
    ├── train_kitti.sh
    └── test_kitti.sh
```

---

## 5. Installation

### 5.1 Clone the Repository

```bash
git clone https://github.com/your-username/BEV-Dynamic-Tracking.git
cd BEV-Dynamic-Tracking
```

### 5.2 Create Environment

Using conda is recommended.

```bash
conda create -n bev_tracking python=3.8 -y
conda activate bev_tracking
```

### 5.3 Install Dependencies

```bash
pip install -r requirements.txt
```

If your implementation depends on CUDA extensions, spconv, mmcv, or mmdet3d, please install the versions compatible with your CUDA and PyTorch environment.

Example:

```bash
pip install torch torchvision torchaudio
pip install numpy scipy tqdm pyyaml easydict scikit-learn opencv-python
```

---

## 6. Dataset Preparation

### 6.1 KITTI Tracking Dataset

Download the KITTI tracking dataset and organize it as follows:

```text
data/kitti/
├── training/
│   ├── calib/
│   ├── label_02/
│   └── velodyne/
└── testing/
    ├── calib/
    └── velodyne/
```

Then update the dataset path in the config file:

```yaml
DATA_CONFIG:
  DATA_PATH: data/kitti
```

### 6.2 nuScenes / Waymo

The framework is designed to be extensible to larger autonomous driving datasets such as nuScenes and Waymo. Please follow the corresponding dataset conversion scripts and update the configuration files before training or evaluation.

---

## 7. Training

### 7.1 Train on KITTI

```bash
python tools/train.py \
  --cfg configs/kitti/car.yaml \
  --batch_size 16 \
  --epochs 80 \
  --workers 4
```

For different object categories:

```bash
python tools/train.py --cfg configs/kitti/car.yaml
python tools/train.py --cfg configs/kitti/van.yaml
python tools/train.py --cfg configs/kitti/pedestrian.yaml
python tools/train.py --cfg configs/kitti/cyclist.yaml
```

### 7.2 Resume Training

```bash
python tools/train.py \
  --cfg configs/kitti/car.yaml \
  --ckpt outputs/checkpoints/checkpoint_epoch_40.pth \
  --resume
```

---

## 8. Evaluation

### 8.1 Test a Trained Model

```bash
python tools/test.py \
  --cfg configs/kitti/car.yaml \
  --ckpt outputs/checkpoints/best_model.pth
```

### 8.2 Evaluation Metrics

Following common 3D single object tracking protocols, we report:

- **Success**: overlap-based tracking accuracy.
- **Precision**: center-distance-based localization precision.

Example output:

```text
*************** Evaluation Results ***************
Success:   0.XXXX
Precision: 0.XXXX
**************************************************
```

---

## 9. Visualization

To visualize tracking results:

```bash
python tools/visualize.py \
  --cfg configs/kitti/car.yaml \
  --ckpt outputs/checkpoints/best_model.pth \
  --sequence 0000
```

The visualization module can be used to inspect:

- Predicted 3D bounding boxes.
- Ground-truth trajectories.
- Tracking drift cases.
- Occlusion and sparse point cloud scenarios.
- Success / failure examples across object categories.

---

## 10. Experimental Results

The current project version reports stable improvements over the BEV baseline on multiple KITTI object categories.

| Category | Success Gain | Precision Gain |
|---|---:|---:|
| Car | +0.7 | +0.9 |
| Van | +0.7 | +2.6 |
| Pedestrian | +0.2 | +0.2 |
| Cyclist | +0.9 | +0.0 |
| **Average** | **+0.5** | **+0.8** |

> Note: Please replace the table above with the final results from your official experiment logs before public release.

---

## 11. Ablation Study

The effectiveness of the proposed modules can be verified by comparing the baseline with TIW, DWM, and the full model.

| Method | TIW | DWM | Success | Precision |
|---|:---:|:---:|---:|---:|
| BEV Baseline | ✗ | ✗ | - | - |
| Baseline + TIW | ✓ | ✗ | - | - |
| Baseline + DWM | ✗ | ✓ | - | - |
| **Baseline + TIW + DWM** | ✓ | ✓ | **Best** | **Best** |

The combination of TIW and DWM improves both tracking continuity and localization stability, showing that temporal interaction and dynamic feature modulation are complementary.

---

## 12. Supported Tasks

- [x] LiDAR-based 3D single object tracking
- [x] BEV feature extraction
- [x] Historical-current frame interaction
- [x] Dynamic feature reweighting
- [x] KITTI training and evaluation
- [x] Result visualization
- [ ] nuScenes full training pipeline
- [ ] Waymo full training pipeline
- [ ] Real-time deployment demo

---

## 13. Roadmap

- [ ] Improve small-object tracking performance for pedestrians and cyclists.
- [ ] Add more robust long-term temporal memory.
- [ ] Extend experiments to nuScenes and Waymo.
- [ ] Provide pretrained checkpoints.
- [ ] Provide complete visualization demo.
- [ ] Optimize inference latency for edge deployment.
- [ ] Release detailed training logs and ablation results.

---

## 14. Team

This project is developed by the undergraduate research team focusing on 3D perception, autonomous driving, and intelligent transportation.

| Name | Role | Responsibility |
|---|---|---|
| 李国玉 | Project Lead | Overall planning, algorithm design, model implementation, experiment analysis |
| 陈赛 | Member | Model development, training optimization, baseline reproduction |
| 何海洋 | Member | Data processing, system testing, experiment validation |
| 王梓涵 | Member | Material organization, visualization design, presentation support |
| 李胜杰 | Advisor | Research guidance, technical direction, project supervision |

---

## 15. Citation

If this repository is helpful for your research or project, please consider citing it:

```bibtex
@misc{bev_dynamic_tracking_2026,
  title        = {BEV Dynamic Tracking: BEV-based 3D Single Object Tracking for Complex Driving Scenarios},
  author       = {Li, Guoyu and Chen, Sai and He, Haiyang and Wang, Zihan},
  year         = {2026},
  howpublished = {\url{https://github.com/your-username/BEV-Dynamic-Tracking}}
}
```

---

## 16. Acknowledgements

This project is built upon prior research in LiDAR-based 3D single object tracking, BEV perception, and autonomous driving. We thank the open-source community for providing valuable datasets, benchmarks, and research codebases.

Representative related topics include:

- LiDAR-based 3D single object tracking
- BEV representation learning
- Point cloud perception
- Temporal modeling for autonomous driving
- Robust tracking under occlusion and sparse observations

---

## 17. License

This project is released under the MIT License. Please refer to `LICENSE` for more details.

---

## 18. Contact

For questions, suggestions, or collaboration, please open an issue or contact the project maintainers.

```text
Project: BEV Dynamic Tracking
Task: LiDAR-based 3D Single Object Tracking
Scenario: Autonomous Driving / Intelligent Transportation / Unmanned Systems
```
