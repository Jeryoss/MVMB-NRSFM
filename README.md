# MVMB-NRSFM
Code for Deep NRSFM for Multi-view Multi-body Pose Estimation.

This paper addresses the challenging task of unsupervised relative human pose estimation. Our solution exploits the potential offered by utilizing multiple (2-5) uncalibrated cameras. It is assumed that
spatial human pose and camera parameter estimation can be solved as a block sparse dictionary learning problem with zero supervision. The resulting structures and camera parameters can fit individual
skeletons into a common space. To do so, we exploit the fact that all individuals in the image are
viewed from the same camera viewpoint, thus exploiting the information provided by multiple camera
views and overcoming the lack of information on camera parameters. To the best of our knowledge,
this is the first solution that requires neither 3D ground truth nor knowledge of the intrinsic or extrinsic camera parameters. Our approach demonstrates the potential of using multiple viewpoints to solve
challenging computer vision problems.

## Installation
1. Using Apptainer
2. Using conda environment

## Usage
1. Training and Testing: main.py
2. inference: inference.py

Note: further documentation will be added soon.