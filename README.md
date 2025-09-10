# Dynamic Integration of Preference and Knowledge Status for Knowledge Concept Recommendation in MOOCs

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
