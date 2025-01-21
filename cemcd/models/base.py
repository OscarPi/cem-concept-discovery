import numpy as np
import torch
import lightning
from cemcd.metrics import calculate_concept_accuracies, calculate_task_accuracy

class BaseModel(lightning.LightningModule):
    def __init__(self, n_tasks, task_class_weights, concept_loss_weights):
        super().__init__()

        self.n_tasks = n_tasks

        self.intervention_mask = None

        self.loss_concept = torch.nn.BCELoss(weight=concept_loss_weights)
        if self.n_tasks > 1:
            self.loss_task = torch.nn.CrossEntropyLoss(weight=task_class_weights)
        else:
            self.loss_task = torch.nn.BCEWithLogitsLoss(weight=task_class_weights)

    def forward(self, x, c_true=None, train=False):
        raise NotImplementedError()

    def run_step(self, batch, train=False):
        x, y, c = batch

        result = self.forward(
            x,
            c_true=c,
            train=train
        )

        predicted_concept_probs = result[0]
        predicted_labels = result[1]

        task_loss = self.loss_task(predicted_labels.squeeze(), y)

        concept_loss = 0
        c_accuracy, c_accuracies, c_auc, c_aucs = np.nan, [np.nan], np.nan, [np.nan]
        if self.concept_loss_weight > 0:
            c_used = torch.where(
                torch.logical_and(c >= 0, c <= 1),
                c,
                torch.zeros_like(c)
            )
            predicted_concept_probs_used = torch.where(
                torch.logical_and(c >= 0, c <= 1),
                predicted_concept_probs,
                torch.zeros_like(predicted_concept_probs)
            )

            concept_loss = self.loss_concept(predicted_concept_probs_used, c_used)
            c_accuracy, c_accuracies, c_auc, c_aucs = calculate_concept_accuracies(predicted_concept_probs, c)

        loss = self.concept_loss_weight * concept_loss + task_loss

        y_accuracy = calculate_task_accuracy(predicted_labels, y)

        result = {
            "c_accuracy": c_accuracy,
            "c_accuracies": c_accuracies,
            "c_auc": c_auc,
            "c_aucs": c_aucs,
            "y_accuracy": y_accuracy
        }

        return loss, result

    def training_step(self, batch, batch_idx):
        loss, result = self.run_step(batch, train=True)
        self.log("loss", float(loss), prog_bar=True)
        self.log("c_accuracy", result["c_accuracy"], prog_bar=True)
        self.log("c_auc", result["c_auc"], prog_bar=True)
        self.log("y_accuracy", result["y_accuracy"], prog_bar=True)
        for i, accuracy in enumerate(result["c_accuracies"]):
            self.log(f"concept_{i+1}_accuracy", accuracy)
        for i, auc in enumerate(result["c_aucs"]):
            self.log(f"concept_{i+1}_auc", auc)
        return {
            "loss": loss,
            "log": {**result, "loss": float(loss)}
        }

    def validation_step(self, batch, batch_idx):
        loss, result = self.run_step(batch)
        self.log("val_loss", float(loss), prog_bar=True)
        self.log("val_c_accuracy", result["c_accuracy"], prog_bar=True)
        self.log("val_c_auc", result["c_auc"], prog_bar=True)
        self.log("val_y_accuracy", result["y_accuracy"], prog_bar=True)
        for i, accuracy in enumerate(result["c_accuracies"]):
            self.log(f"concept_{i+1}_val_accuracy", accuracy)
        for i, auc in enumerate(result["c_aucs"]):
            self.log(f"concept_{i+1}_val_auc", auc)
        return {
            "val_" + key: val for key, val in list(result.items()) + [("loss", float(loss))]
        }

    def test_step(self, batch, batch_idx):
        loss, result = self.run_step(batch)
        self.log("test_loss", float(loss), prog_bar=True)
        self.log("test_c_accuracy", result["c_accuracy"], prog_bar=True)
        self.log("test_c_auc", result["c_auc"], prog_bar=True)
        self.log("test_y_accuracy", result["y_accuracy"], prog_bar=True)
        for i, accuracy in enumerate(result["c_accuracies"]):
            self.log(f"concept_{i+1}_test_accuracy", accuracy)
        for i, auc in enumerate(result["c_aucs"]):
            self.log(f"concept_{i+1}_test_auc", auc)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y, c = batch
        return self.forward(
            x,
            c_true=c
        )

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
