import torch
import lightning

class LinearModel(lightning.LightningModule):
    def __init__(self, in_dim, pos_weight=1.0):
        super().__init__()

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 1),
        )

        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, x):
        return self.predictor(x).squeeze()

    def run_step(self, batch):
        x, y = batch

        result = self.forward(x)

        loss = self.loss(result, y)

        return loss

    def training_step(self, batch, _):
        loss = self.run_step(batch)
        self.log("loss", float(loss), prog_bar=True)
        return {
            "loss": loss,
            "log": {"loss": float(loss)}
        }

    def validation_step(self, batch, _):
        loss = self.run_step(batch)
        self.log("val_loss", float(loss), prog_bar=True)
        return {
            "val_loss": float(loss)
        }

    def test_step(self, batch, _):
        loss = self.run_step(batch)
        self.log("test_loss", float(loss), prog_bar=True)
        return loss

    def predict_step(self, batch, _):
        x, _ = batch
        return self.forward(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }
