#!/usr/bin/env bash

# conda init 注意conda路径 这里是miniconda
source ~/miniconda3/etc/profile.d/conda.sh
# 激活自己的环境 我这里是myenv
conda activate my_radar

# 切换到当前文件所在目录
cd $(dirname $0)
# 挂载数据集 (现在的训练环境挂载路径有问题，会导致与开发环境不一致，所以需要再连接一次)
ln -s /ai/506040ad32d6/kitti /ai/datasets/kitti

# 训练命令
# scripts/dist_train.sh 为 OpenPCDet的多卡训练脚本
scripts/dist_train.sh 4 --cfg_file cfgs/kitti_models/second.yaml --extra_tag baseline