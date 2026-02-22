# AAAI 2024: A Twist for Graph Classification: Optimizing Causal Information Flow in Graph Neural Networks
This repository contains code for the AAAI 2024 paper: [A Twist for Graph Classification: Optimizing Causal Information Flow in Graph Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/29648)


## Dependencies
Please setup the environment with Python 3.10 and CUDA 11.8. Typically, you might need to run the following commands:

```
pip install --upgrade pip setuptools wheel
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2+cu118 \
--index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
-f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install networkx matplotlib
```

## Experiments

### For dir datasets 
```
python main_dir.py --dataset mnist --bias 0.8 --model 'CausalGCN'  --train_model swl
python main_dir.py --dataset mnist --bias 0.85 --model 'CausalGCN'  --train_model mgda 
```
### For TU datasets

```
python main_real.py --model CausalGAT --dataset MUTAG --train_model swl
python main_real.py --model CausalGAT --dataset MUTAG --train_model mgda 
```

### For syn datasets

```
python main_syn.py --model CausalGAT --bias 0.7 --train_model swl
python main_syn.py --model CausalGAT --bias 0.7 --train_model mgda 
```

## Data download
dir datasets can be get in my paper [dir_data_geter](https://github.com/haibin65535/temp/tree/main/dir_data_geter),
TU datasets and syn datasets can be downloaded when you run ``main_real.py`` and ``main_syn.py``.

```
@article{Zhao_Wang_Wen_Zhang_Zhou_Wang_2024,
title={A Twist for Graph Classification: Optimizing Causal Information Flow in Graph Neural Networks},
volume={38}, url={https://ojs.aaai.org/index.php/AAAI/article/view/29648},
DOI={10.1609/aaai.v38i15.29648},
number={15},
journal={Proceedings of the AAAI Conference on Artificial Intelligence},
author={Zhao, Zhe and Wang, Pengkun and Wen, Haibin and Zhang, Yudong and Zhou, Zhengyang and Wang, Yang},
year={2024}, month={Mar.}, pages={17042-17050} }
```



