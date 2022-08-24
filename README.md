# Distributed DeepGCNs

简体中文 | [English](README_en.md)

本项目在幻方萤火超算集群上对 [DeepGCNs](https://github.com/lightaime/deep_gcns_torch) 模型进行多卡并行计算优化，解决大规模图计算场景下的多卡加速问题，实现训练效率的极大提升。

DeepGCNs 借鉴 CNN 的概念，可以实现很深层结构的 GNN 模型训练，以在各种图任务中取得优异的泛化能力。其主要是残差/密集连接和扩张卷积，并将它们适应于 GCN 架构。

相关论文：
* DeepGCNs ([ICCV'2019](https://arxiv.org/abs/1904.03751), [TPAMI'2021](https://arxiv.org/abs/1910.06849))
* DeeperGCN ([Arxiv'2020](https://arxiv.org/abs/2006.07739))
* GNN'1000 ([ICML'2021](https://arxiv.org/abs/2106.07476))


## Requirements
* [hfai](https://doc.hfai.high-flyer.cn/index.html)
* [Pytorch>=1.8.0](https://pytorch.org)
* [pytorch_geometric>=1.6.0](https://pytorch-geometric.readthedocs.io/en/latest/)

## Training

原始数据来自斯坦福大学提供的一个图综合数据集 [Open Graph Benchmark](https://ogb.stanford.edu/) ，幻方AI将其进行了整理，合入 `hfai.datasets` 数据集仓库中，提供高速数据读取接口，使用参考 [hfai文档](https://doc.hfai.high-flyer.cn/api/datasets.html#hfai.datasets.OGB) 。

1. 点属性预测（`ogbn-proteins`）

   提交任务至萤火集群
   ```shell
    hfai python ogbn_proteins/main.py -- -n 1
   ```
   本地运行：
   ```shell
    python ogbn_proteins/main.py
   ```

2. 边链接预测（`ogbl-collab`）

    提交任务至萤火集群
   ```shell
    hfai python ogbl_collab/main.py -- -n 1
   ```
   本地运行：
   ```shell
    python ogbl_collab/main.py
   ```

3. 图属性预测（`ogbg-ppa`）

    提交任务至萤火集群
   ```shell
    hfai python ogbg_ppa/main.py -- -n 1
   ```
   本地运行：
   ```shell
    python ogbg_ppa/main.py
   ```