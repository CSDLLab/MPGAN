# MPGAN

Implementation of the paper **Using Simulated Training Data of Voxel-level Generative Model to Improve 3D Neuron Reconstruction**. 

## Introduction

In this paper, we propose a novel strategy of using two-stage generative models to simulate training data with voxel-level labels (MPGAN). Trained upon unlabeled data by optimizing a novel objective function of preserving predefined labels, the models are able to synthesize realistic 3D images with underlying voxel labels. The framework of MPGAN model is shown as follow:

![image-20220318174040337](imgs/1.png)	

## Getting Started

### Prerequisites

* Linux 
* NVIDIA GPU
* TensorFlow==1.15.0

### Installation

Create virtual environment and install required packages:

```python
conda create -n mpgan
conda activate mpgan
conda install --yes --file requirements.txt
```

## Usage

### First Stage: Neuron Image Simulator 

```python
python 1_simulator.py
```

<img src="imgs/2.png" alt="image-20220318174040337"  />	

### Second Stage: GAN

```python
python 2_mpgan.py
```

## Data



![image-20220318174040337](imgs/3.png)	




## Reference