# Distributed DeepGCNs

English | [简体中文](README.md)

This is a distributed [DeepGCNs](https://github.com/lightaime/deep_gcns_torch) model implemented and improved by High-Flyer AI. It can use multiple GPUs to achieve training acceleration.

DeepGCNs borrow concepts from CNNs, and mainly adapt residual/dense connections and dilated convolutions to GCN architectures. 

Reference papers:
* DeepGCNs ([ICCV'2019](https://arxiv.org/abs/1904.03751), [TPAMI'2021](https://arxiv.org/abs/1910.06849))
* DeeperGCN ([Arxiv'2020](https://arxiv.org/abs/2006.07739))
* GNN'1000 ([ICML'2021](https://arxiv.org/abs/2106.07476))

## Requirements
* [hfai](https://doc.hfai.high-flyer.cn/index.html)
* [Pytorch>=1.8.0](https://pytorch.org)
* [pytorch_geometric>=1.6.0](https://pytorch-geometric.readthedocs.io/en/latest/)

## Training
The raw data is from the public dataset, [Open Graph Benchmark](https://ogb.stanford.edu/) , which is integrated into the high-flyer dataset warehouse, [`hfai.datasets`](https://doc.hfai.high-flyer.cn/api/datasets.html#hfai.datasets.OGB).

1. Node Property Prediction (`ogbn-proteins`)

   submit the task to Yinghuo HPC:
   ```shell
    hfai python ogbn_proteins/main.py -- -n 1
   ```
   run locally:
   ```shell
    python ogbn_proteins/main.py
   ```

2. Link Property Prediction (`ogbl-collab`)

    submit the task to Yinghuo HPC:
   ```shell
    hfai python ogbl_collab/main.py -- -n 1
   ```
   run locally:
   ```shell
    python ogbl_collab/main.py
   ```

3. Graph Property Prediction (`ogbg-ppa`)

    submit the task to Yinghuo HPC:
   ```shell
    hfai python ogbg_ppa/main.py -- -n 1
   ```
   run locally:
   ```shell
    python ogbg_ppa/main.py
   ```
