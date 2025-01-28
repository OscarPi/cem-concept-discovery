import numpy as np
import torch
import lightning
from cemcd.metrics import calculate_task_accuracy

class BlackBoxModel(lightning.LightningModule):
    def __init__(self, n_tasks, latent_representation_size, task_class_weights=None):
        super().__init__()

        self.n_tasks = n_tasks
        self.latent_representation_size = latent_representation_size

        if self.n_tasks > 1:
            self.loss_task = torch.nn.CrossEntropyLoss(weight=task_class_weights)
        else:
            self.loss_task = torch.nn.BCEWithLogitsLoss(weight=task_class_weights)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(latent_representation_size, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, n_tasks)
        )

    def forward(self, x):
        predicted_labels = self.model(x)
        return predicted_labels


    def run_step(self, batch):
        x, y, _ = batch

        predicted_labels = self.forward(x)

        loss = self.loss_task(predicted_labels.squeeze(), y)

        y_accuracy = calculate_task_accuracy(predicted_labels, y)

        result = {
            "y_accuracy": y_accuracy
        }

        return loss, result

    def training_step(self, batch, batch_idx):
        loss, result = self.run_step(batch)
        self.log("loss", float(loss), prog_bar=True)
        self.log("y_accuracy", result["y_accuracy"], prog_bar=True)
        return {
            "loss": loss,
            "log": {**result, "loss": float(loss)}
        }

    def validation_step(self, batch, batch_idx):
        loss, result = self.run_step(batch)
        self.log("val_loss", float(loss), prog_bar=True)
        self.log("val_y_accuracy", result["y_accuracy"], prog_bar=True)
        return {
            "val_" + key: val for key, val in list(result.items()) + [("loss", float(loss))]
        }

    def test_step(self, batch, batch_idx):
        loss, result = self.run_step(batch)
        self.log("test_loss", float(loss), prog_bar=True)
        self.log("test_y_accuracy", result["y_accuracy"], prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, _, _ = batch
        return self.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }
