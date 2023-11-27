# Interaction Generalization

## Install the Environment

``` bash
conda create -n roach python=3.7
conda activate roach

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install opencv-python
pip install ujson
pip install shapely
pip install networkx
pip install gym==0.17.2
pip install h5py
pip install hydra-core
pip install wandb
pip install pygame


# cd PythonAPI/carla/dist
pip install carla-0.9.14-cp37-cp37m-manylinux_2_27_x86_64.whl
```