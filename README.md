# CEMCD
---
A library for discovering concepts encoded in the latent space of CEMs and using the discovered concepts to train more interpretable models.

## Credits
---
Much of the code in this repository is based on the [implementation](https://github.com/mateoespinosa/cem) released by Espinosa Zarlenga et al. for their NeurIPS 2022 paper "Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off".

## Installation
---

Make sure you have Python 3.10 and pip >= 24.0 installed, then run:

`
    pip install --editable .
`

## Usage
---
`
    python experiments/run_experiment.py <experiment_config.yaml>
`

Before running an experiment you need to download the dataset. MNIST will be downloaded automatically. dSprites can be downloaded from [here](https://github.com/google-deepmind/dsprites-dataset). Animals with Attributes 2 can be downloaded [here](https://cvml.ista.ac.at/AwA2/). Then run `generate_awa_splits.py` to produce train, validation and test splits. To download CUB, follow the instructions [here](https://github.com/yewsiang/ConceptBottleneck/tree/master/CUB).
