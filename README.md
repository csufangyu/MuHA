# MuHA
**An Effective Framework with Hybrid Augmentation for Visual Reinforcement Learning Generalization**

This is the official implementation of **An Effective Framework with Hybrid Augmentation for Visual Reinforcement Learning Generalization**

## Setup

The MuJoCo license and instruction can be found at https://github.com/deepmind/mujoco;The Mujoco Version should be `2.1.1` for training.

The DeepMind Control license and instruction can be found at https://github.com/deepmind/dm_control

For training, the dependencies are based on DrQ-v2. You can install them with the following commands:

```
conda env create -f conda_env.yml
```
Detailed installation instructions can be found at: https://github.com/facebookresearch/drqv2

For generalization testing, we use the DMControl Gneralization Benchmark.  You can run the commands as follows:

```
cd dmcontrol-generalization-benchmark/
conda env create -f setup/dmcgb.yml
conda activate dmcgb
sh setup/install_envs.sh
```

The place365 dataset can be downloaded by running:
```
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
```
After downloading and extracting the data, add your dataset directory to the `config.cfg`. 

Detailed installation instructions can be found at: https://github.com/nicklashansen/dmcontrol-generalization-benchmark

## Training and Evaluation

### Training

`muha` conda environment is served for training, so you should activate this conda env at first:

```
conda activate muha
bash train.sh
```
`cd` to the `exp_local` file and move the trained model to the test file:
```
mv snapshot.pt dmcontrol-generalization-benchmark/logs/cartpole_swingup/muha/1
```

### Evaluation

```
cd ~/PIEG/dmcontrol-generalization-benchmark/
conda activate dmcgb
bash script/eval/muha.sh
```
## Acknowledgement

Our code is generally built upon: https://github.com/gemcollector/PIE-G