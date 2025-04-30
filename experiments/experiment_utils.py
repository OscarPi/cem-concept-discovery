from pathlib import Path
import yaml
import torch
import lightning
from torchvision.models import resnet34
from cemcd.models.pre_concept_models import get_pre_concept_model
from cemcd.data import awa, cub, dsprites, mnist, celeba, shapes, cifar100
from cemcd.training import train_cem, load_cem

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
    elif config["dataset"] == "dsprites":
        datasets = []
        for foundation_model in config["foundation_models"]:
            print(f"Running foundation model {foundation_model}.")
            datasets.append(dsprites.DSpritesDatasets(
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
    elif config["dataset"] == "celeba":
        datasets = []
        for foundation_model in config["foundation_models"]:
            print(f"Running foundation model {foundation_model}.")
            datasets.append(celeba.CELEBADatasets(
                foundation_model=foundation_model,
                dataset_dir=config["dataset_dir"],
                model_dir=config["model_dir"]))
        return datasets
    elif config["dataset"] == "cifar100":
        datasets = []
        for foundation_model in config["foundation_models"]:
            print(f"Running foundation model {foundation_model}.")
            datasets.append(cifar100.CIFARDatasets(
                foundation_model=foundation_model,
                dataset_dir=config["dataset_dir"],
                model_dir=config["model_dir"]))
        return datasets
    raise ValueError(f"Unrecognised dataset: {config['dataset']}")

def make_pre_concept_model(config):
    if config["pre_concept_model"] == "cnn":
        cnn_config = config["pre_concept_cnn_config"]
        return get_pre_concept_model(cnn_config["width"], cnn_config["height"], cnn_config["channels"])
    elif config["pre_concept_model"] == "resnet34":
        return resnet34(pretrained=True)

    raise ValueError(f"Unknown pre concept model: {config['pre_concept_model']}")

def make_concept_model(config, n_concepts):
    if config["pre_concept_model"] == "cnn":
        cnn_config = config["pre_concept_cnn_config"]
        return get_pre_concept_model(cnn_config["width"], cnn_config["height"], cnn_config["channels"], n_concepts)
    elif config["pre_concept_model"] == "resnet34":
        return torch.nn.Sequential(resnet34(pretrained=True), torch.nn.Linear(1000, n_concepts))

    raise ValueError(f"Unknown pre concept model: {config['pre_concept_model']}")

def get_initial_models(config, datasets, run_dir):
    models = []
    test_results = []
    for dataset in datasets:
        if config.get("cache_dir", None) is not None:
            load_path = Path(config["cache_dir"]) / f"initial_{dataset.foundation_model or 'basic'}cem.pth"
            print(f"Loading model from {load_path}.")
            model, test_result = load_cem(
                n_concepts=dataset.n_concepts,
                n_tasks=dataset.n_tasks,
                pre_concept_model=None,
                latent_representation_size=dataset.latent_representation_size,
                train_dl=dataset.train_dl(),
                test_dl=dataset.test_dl(),
                path=load_path,
                use_task_class_weights=config["use_task_class_weights"],
                use_concept_loss_weights=config["use_concept_loss_weights"])
        else:
            if run_dir is None:
                save_path = None
            else:
                save_path = Path(run_dir) / f"initial_{dataset.foundation_model or 'basic'}cem.pth"
            model, test_result = train_cem(
                n_concepts=dataset.n_concepts,
                n_tasks=dataset.n_tasks,
                pre_concept_model=None,
                latent_representation_size=dataset.latent_representation_size,
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

    # c_pred, _, _ = cemcd.concept_discovery.calculate_embeddings(model, train_dl)
    # model.intervention_on_value = torch.from_numpy(np.percentile(c_pred, 95, axis=0))
    # model.intervention_off_value = torch.from_numpy(np.percentile(c_pred, 5, axis=0))
    # print(f"{model_name} interventions on value: {model.intervention_on_value}")
    # print(f"{model_name} interventions off value: {model.intervention_off_value}")

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
