from pathlib import Path
import yaml
import torch
import lightning
from cemcd.data import awa, cub, mnist, shapes, kitchens
from cemcd.training import train_cem

def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def load_datasets(config):
    if config["dataset"] == "mnist_add":
        mnist_config = config["mnist_config"]
        datasets = []
        for foundation_model in config["foundation_models"]:
            print(f"Running foundation model {foundation_model}.")
            datasets.append(mnist.MNISTDatasets(
                n_digits=mnist_config["n_digits"],
                max_digit=mnist_config["max_digit"],
                foundation_model=foundation_model,
                dataset_dir=config["dataset_dir"],
                model_dir=config["model_dir"]))
        return datasets
    elif config["dataset"] == "shapes":
        datasets = []
        for foundation_model in config["foundation_models"]:
            print(f"Running foundation model {foundation_model}.")
            datasets.append(shapes.ShapesDatasets(
                foundation_model=foundation_model,
                dataset_dir=config["dataset_dir"],
                model_dir=config["model_dir"]))
        return datasets
    elif config["dataset"] == "cub":
        datasets = []
        for foundation_model in config["foundation_models"]:
            print(f"Running foundation model {foundation_model}.")
            datasets.append(cub.CUBDatasets(
                foundation_model=foundation_model,
                dataset_dir=config["dataset_dir"],
                model_dir=config["model_dir"]))
        return datasets
    elif config["dataset"] == "awa":
        datasets = []
        for foundation_model in config["foundation_models"]:
            print(f"Running foundation model {foundation_model}.")
            datasets.append(awa.AwADatasets(
                foundation_model=foundation_model,
                dataset_dir=config["dataset_dir"],
                model_dir=config["model_dir"]))
        return datasets
    elif config["dataset"] == "kitchens":
        datasets = []
        for foundation_model in config["foundation_models"]:
            print(f"Running foundation model {foundation_model}.")
            datasets.append(kitchens.KitchensDatasets(
                foundation_model=foundation_model,
                dataset_dir=config["dataset_dir"],
                model_dir=config["model_dir"]))
        return datasets
    raise ValueError(f"Unrecognised dataset: {config['dataset']}")

def train_initial_cems(config, datasets, run_dir):
    models = []
    test_results = []
    for dataset in datasets:
        if run_dir is None:
            save_path = None
        else:
            save_path = Path(run_dir) / f"initial_{dataset.foundation_model}_cem.pth"
        model, test_result = train_cem(
            n_concepts=dataset.n_concepts,
            n_tasks=dataset.n_tasks,
            latent_representation_size=dataset.latent_representation_size,
            embedding_size=config["cem_embedding_size"],
            concept_loss_weight=config["cem_concept_loss_weight"],
            train_dl=dataset.train_dl(),
            val_dl=dataset.val_dl(),
            test_dl=dataset.test_dl(),
            save_path=save_path,
            max_epochs=config["max_epochs"],
            use_task_class_weights=config["use_task_class_weights"],
            use_concept_loss_weights=config["use_concept_loss_weights"])
        models.append(model)
        test_results.append(test_result)
    return models, test_results

def get_intervention_accuracies(model, test_dl, concepts_to_intervene, one_at_a_time):
    trainer = lightning.Trainer()

    intervention_accuracies = []
    model.intervention_mask = torch.tensor([0] * model.n_concepts)
    [test_results] = trainer.test(model, test_dl)
    initial_task_accuracy = test_results["test_y_accuracy"]
    for c in concepts_to_intervene:
        if one_at_a_time:
            model.intervention_mask = torch.tensor([0] * model.n_concepts)
        model.intervention_mask[c] = 1
        [test_results] = trainer.test(model, test_dl)
        accuracy_difference = round(test_results["test_y_accuracy"] - initial_task_accuracy, 4)
        intervention_accuracies.append(accuracy_difference)
    return intervention_accuracies
