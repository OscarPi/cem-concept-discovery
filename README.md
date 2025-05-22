# HiCEMs and Concept Splitting

A library for discovering sub-concepts encoded in the latent space of CEMs and using the discovered concepts to train more interpretable models.

## Credits

Some of the code in this repository is based on the [implementation](https://github.com/mateoespinosa/cem) released by Espinosa Zarlenga et al. for their NeurIPS 2022 paper "Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off".

## Installation


Make sure you have Python 3.10 and pip >= 24.0 installed.
Install [CLIP](https://github.com/openai/CLIP), then run:

`
    pip install --editable .
`

## Usage

`
    python experiments/run_experiment.py <experiment_config.yaml>
`

or

`
    python experiments/run_baselines.py <experiment_config.yaml>
`

Before running an experiment, you need to download the dataset, and update the `dataset_dir`, `model_dir` and `results_dir` in the experiment config file. (The `model_dir` and `results_dir` can be empty: any models should be downloaded automatically.)

 MNIST will be downloaded automatically. To use the shapes dataset, place [shapes_dataset.pkl](shapes_dataset.pkl) in a directory called `shapes` inside the `dataset_dir` specified in the experiment config file. Animals with Attributes 2 can be downloaded [here](https://cvml.ista.ac.at/AwA2/). Unzip the dataset and place it in a directory called `AwA2` inside the `dataset_dir` specified in the experiment config file. Then copy the contents of [splits/AwA2](splits/AwA2) to the `AwA2` directory. CUB can be downloaded [here](http://www.vision.caltech.edu/datasets/cub_200_2011/). Extract it and place it in a directory called `CUB` inside the `dataset_dir` specified in the experiment config file. Copy the contents of [splits/CUB](splits/CUB) to the `CUB` directory.
