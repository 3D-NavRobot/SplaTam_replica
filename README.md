<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM</h1>
  

  <h3 align="center"><a href="https://arxiv.org/pdf/2312.02126.pdf">Paper</a> | <a href="https://youtu.be/jWLI-OFp3qU">Video</a> | <a href="https://spla-tam.github.io/">Project Page</a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./assets/1.gif" alt="Logo" width="100%">
  </a>
</p>

<br>

## Adopted from SplaTam paper 


## Installation

##### (Recommended)
SplaTAM has been benchmarked with Python 3.10, Torch 1.12.1 & CUDA=11.6. However, Torch 1.12 is not a hard requirement and the code has also been tested with other versions of Torch and CUDA such as Torch 2.3.0 & CUDA 12.1.

The simplest way to install all dependences is to use [anaconda](https://www.anaconda.com/) and [pip](https://pypi.org/project/pip/) in the following steps: 

```bash
conda create -n splatam python=3.10
conda activate splatam
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```

<!-- Alternatively, we also provide a conda environment.yml file :
```bash
conda env create -f environment.yml
conda activate splatam
``` -->

#### Windows

For installation on Windows using Git bash, please refer to the [instructions shared in Issue#9](https://github.com/spla-tam/SplaTAM/issues/9#issuecomment-1848348403).

#### Docker and Singularity Setup

We also provide a docker image. We recommend using a venv to run the code inside a docker image:


```bash
docker pull nkeetha/splatam:v1
bash bash_scripts/start_docker.bash
cd /SplaTAM/
pip install virtualenv --user
mkdir venv
cd venv
virtualenv --system-site-packages splatam
source ./splatam/bin/activate
pip install -r venv_requirements.txt
```

Setting up a singularity container is similar:
```bash
cd </path/to/singularity/folder/>
singularity pull splatam.sif docker://nkeetha/splatam:v1
singularity instance start --nv splatam.sif splatam
singularity run --nv instance://splatam
cd <path/to/SplaTAM/>
pip install virtualenv --user
mkdir venv
cd venv
virtualenv --system-site-packages splatam
source ./splatam/bin/activate
pip install -r venv_requirements.txt
```

## Demo

## Benchmarking

For running SplaTAM, we recommend using [weights and biases](https://wandb.ai/) for the logging. This can be turned on by setting the `wandb` flag to True in the configs file. Also make sure to specify the path `wandb_folder`. If you don't have a wandb account, first create one. Please make sure to change the `entity` config to your wandb account. Each scene has a config folder, where the `input_folder` and `output` paths need to be specified. 

Below, we show some example run commands for one scene from each dataset. After SLAM, the trajectory error will be evaluated along with the rendering metrics. The results will be saved to `./experiments` by default.

### Replica

To run SplaTAM-S on the `room0` scene, run the following command:

```bash
python scripts/splatam.py configs/replica/splatam_s.py
```



## Acknowledgement

We thank the authors of the following repositories for their open-source code:

- 3D Gaussians
  - [Dynamic 3D Gaussians](https://github.com/JonathonLuiten/Dynamic3DGaussians)
  - [3D Gaussian Splating](https://github.com/graphdeco-inria/gaussian-splatting)
- Dataloaders
  - [GradSLAM & ConceptFusion](https://github.com/gradslam/gradslam/tree/conceptfusion)
- Baselines
  - [Nice-SLAM](https://github.com/cvg/nice-slam)
  - [Point-SLAM](https://github.com/eriksandstroem/Point-SLAM)

## Citation

The Baseline paper SplaTam can be cited using:

```bib
@inproceedings{keetha2024splatam,
        title={SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM},
        author={Keetha, Nikhil and Karhade, Jay and Jatavallabhula, Krishna Murthy and Yang, Gengshan and Scherer, Sebastian and Ramanan, Deva and Luiten, Jonathon},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        year={2024}
      }
```

## Developers
- [Nik-V9](https://github.com/Nik-V9) ([Nikhil Keetha](https://nik-v9.github.io/))
- [JayKarhade](https://github.com/JayKarhade) ([Jay Karhade](https://jaykarhade.github.io/))
- [JonathonLuiten](https://github.com/JonathonLuiten) ([Jonathan Luiten](https://www.vision.rwth-aachen.de/person/216/))
- [krrish94](https://github.com/krrish94) ([Krishna Murthy Jatavallabhula](https://krrish94.github.io/))
- [gengshan-y](https://github.com/gengshan-y) ([Gengshan Yang](https://gengshan-y.github.io/))
