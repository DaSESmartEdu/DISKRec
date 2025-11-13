# Dynamic Integration of Preference and Knowledge Status for Knowledge Concept Recommendation (DISKRec)

This repository provides the official implementation for our paper "**Dynamic Integration of Preference and Knowledge Status for Knowledge Concept Recommendation**", published in [Neurocomputing](\url{https://www.sciencedirect.com/science/article/pii/S0925231225024580}). 

DISKRec is a dynamic knowledge concept recommendation model that jointly models learners' preference and knowledge status from learning and assessment behaviors to better capture their evolving learning motivations.

![(a) The overall architecture of DISKRec, which consists of two modules: Status Disentanglement using Dual-DGNNs, Status Integration and Recommendation. (b) The steps in Dual-DGNNs take the initial graph status and a batch of interactions as input, and output the final graph status after three sub-modules: (1) Interaction Encoder, (2) Dynamic Status Updater with Dual-state Integration, (3) Neighbor Propagator.](./Framework.png)

## Environments
<ul>
<li>torch=2.1.1+cu118</li>
<li>torch-scatter=2.1.2+pt21cu118</li>
<li>torch_geometric=2.5.3</li>
</ul>

## Code structure
```
DISKRec-anonymous
├── data
│   ├── model_input
│   │   ├── ednet
│   │   └── mooccubex
│   └── model_output
├── evaluation
│   └── metric.py
├── model
│   ├── combiner.py
│   ├── decayer.py
│   ├── dgnn.py
|   ├── diskrec.py
|   ├── edge_message.py
|   ├── graph.py
|   ├── iterative_updater.py
|   ├── neighbor_algorithm.py
│   └── rate_layer.py
├── util
│   ├── data_util.py
│   ├── early_stop.py
│   ├── global_config.py
│   └── temporal_interations.py
├── README.md
└── run_diskrec.py
```

## Dataset
We provide the dataset to validate the effectiveness of our method.

## Train and Test
Please unzip the dataset before running the code.
```shell
unzip data.zip
```

Then, run the model using the following command.
```shell
python run_diskrec.py
```

## Citation
If you find our repo useful, please consider citing:
```
@article{liang2025DISKRec, 
  title={Dynamic integration of preference and knowledge status for knowledge concept recommendation},
  author={Liang, Qingqing and Lu, Xuesong and Wang, Chunyang and Qian, Weining and Zhou, Aoying},
  journal={Neurocomputing},
  pages={131786},
  year={2025},
  publisher={Elsevier}
}
```

For any questions or clarifications, please contact: qqliang.dase@stu.ecnu.edu.cn