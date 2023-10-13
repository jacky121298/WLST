# Installation

### Requirements

All the codes are tested in the following environment:
* Linux (tested on Ubuntu 14.04/16.04)
* Python 3.6+
* PyTorch 1.1
* CUDA 10.0
* [`spconv v1.0 (commit 8da6f96)`](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634) or [`spconv v1.2`](https://github.com/traveller59/spconv)

### Install `pcdet v0.3`

NOTE: Please re-install `pcdet v0.3` by running `python setup.py develop` even if you have already installed previous version.

a. Clone this repository.

```shell
git clone https://github.com/jacky121298/WLST.git
```

b. Install the dependent libraries as follows:

* Install the dependent python libraries:

```
pip install -r requirements.txt
```

c. Install this `pcdet` library by running the following command:

```shell
python setup.py develop
```

<!-- ### Environment

```shell
conda create --name wlst python=3.6
conda activate wlst
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

# Add WLST to PYTHONPATH by adding the following line to ~/.bashrc (change the path accordingly)
export PYTHONPATH="${PYTHONPATH}:PATH_TO_THIS_REPO"

# Set the cuda path (change the path to your own cuda location)
export CUDA_PATH=/usr/local/cuda-10.0
export CUDA_HOME=/usr/local/cuda-10.0
export PATH=/usr/local/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

# Build spconv
git clone https://github.com/traveller59/spconv.git --recursive
cd spconv && git checkout 7342772
python setup.py bdist_wheel
cd ./dist && pip install *
``` -->