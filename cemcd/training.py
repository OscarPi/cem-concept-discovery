import numpy as np
import torch
import lightning
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from cemcd.models.cem import ConceptEmbeddingModel
from cemcd.models.cbm import ConceptBottleneckModel

def calculate_task_class_weights(n_tasks, train_dl):
    attribute_count = np.zeros((max(n_tasks, 2),))
    samples_seen = 0
    for _, data in enumerate(train_dl):
        (_, y, _) = data
        if n_tasks > 1:
            y = torch.nn.functional.one_hot(
                y,
                num_classes=n_tasks,
            ).cpu().detach().numpy()
        else:
            y = torch.cat(
                [torch.unsqueeze(1 - y, dim=-1), torch.unsqueeze(y, dim=-1)],
                dim=-1,
            ).cpu().detach().numpy()
        attribute_count += np.sum(y, axis=0)
        samples_seen += y.shape[0]
    print("Class distribution is:", attribute_count / samples_seen)
    if n_tasks > 1:
        task_class_weights = samples_seen / attribute_count - 1
    else:
        task_class_weights = np.array(
            [attribute_count[0]/attribute_count[1]]
        )

    return torch.FloatTensor(task_class_weights)

def calculate_concept_loss_weights(n_concepts, train_dl):
    attribute_count = np.zeros((n_concepts,))
    samples_seen = 0
    for _, data in enumerate(train_dl):
        (_, _, c) = data
        c = c.cpu().detach().numpy()
        c = np.nan_to_num(c)
        attribute_count += np.sum(c, axis=0)
        samples_seen += c.shape[0]
    attribute_count[attribute_count == 0] = 1
    imbalance = samples_seen / attribute_count - 1

    return torch.FloatTensor(imbalance)

def train_cem(
        n_concepts,
        n_tasks,
        pre_concept_model,
        latent_representation_size,
        train_dl,
        val_dl,
        test_dl,
        save_path=None,
        max_epochs=300):
    model = ConceptEmbeddingModel(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        pre_concept_model=pre_concept_model,
        latent_representation_size=latent_representation_size,
        task_class_weights=calculate_task_class_weights(n_tasks, train_dl),
        concept_loss_weights=calculate_concept_loss_weights(n_concepts, train_dl)
    )

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=5,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.0,
                patience=15,
                verbose=False,
                mode="min",
            ),
        ],
    )

    trainer.fit(model, train_dl, val_dl)

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    model.freeze()
    [test_results] = trainer.test(model, test_dl)

    return model, test_results

def load_cem(
        n_concepts,
        n_tasks,
        pre_concept_model,
        latent_representation_size,
        train_dl,
        test_dl,
        path):
    model = ConceptEmbeddingModel(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        pre_concept_model=pre_concept_model,
        latent_representation_size=latent_representation_size,
        task_class_weights=calculate_task_class_weights(n_tasks, train_dl),
        concept_loss_weights=calculate_concept_loss_weights(n_concepts, train_dl)
    )

    trainer = lightning.Trainer()

    model.load_state_dict(torch.load(path))

    model.freeze()
    [test_results] = trainer.test(model, test_dl)

    return model, test_results

def train_cbm(
        n_concepts,
        n_tasks,
        concept_model,
        train_dl,
        val_dl,
        test_dl,
        black_box=False,
        save_path=None,
        max_epochs=300):
    concept_loss_weights = None
    if not black_box:
        concept_loss_weights = calculate_concept_loss_weights(n_concepts, train_dl)

    model = ConceptBottleneckModel(
        n_concepts=n_concepts,
        n_tasks=n_tasks,
        concept_model=concept_model,
        task_class_weights=calculate_task_class_weights(n_tasks, train_dl),
        concept_loss_weights=concept_loss_weights,
        black_box=black_box
    )

    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=5,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                min_delta=0.0,
                patience=25,
                verbose=False,
                mode="min",
            ),
        ],
    )

    trainer.fit(model, train_dl, val_dl)

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    model.freeze()
    [test_results] = trainer.test(model, test_dl)

    return model, test_results
