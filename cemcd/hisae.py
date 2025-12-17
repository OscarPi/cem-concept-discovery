import torch
from torch import optim, nn
import lightning

class HierarchicalSparseAutoEncoder(lightning.LightningModule):
    def __init__(self, input_dim):
        super().__init__()
        self.dictionary_size = 6144 # 12288
        self.k = 16
        self.sub_dictionary_size = 512
        self.sub_dictionary_k = 4

        self.top_level_encoder = nn.Sequential(
            nn.Linear(input_dim, self.dictionary_size),
            nn.ReLU(),
        )
        self.sub_encoders = nn.ModuleList()
        for _ in range(self.dictionary_size):
            self.sub_encoders.append(nn.Sequential(
                nn.Linear(input_dim, self.sub_dictionary_size),
                nn.ReLU()
            ))

        self.top_level_decoder = nn.Linear(self.dictionary_size, input_dim)
        self.sub_decoders = nn.ModuleList()
        for _ in range(self.dictionary_size):
            self.sub_decoders.append(nn.Linear(self.sub_dictionary_size, input_dim))

        self.l2_loss = nn.MSELoss()

    def forward(self, x):
        batch_size = x.shape[0]

        active_concepts = torch.full((batch_size, self.k + self.k * self.sub_dictionary_k), -1, dtype=torch.long, device=self.device)

        top_level_activations = self.top_level_encoder(x)

        topk_values, topk_indices = torch.topk(top_level_activations, k=self.k, dim=1)

        top_level_active = torch.zeros_like(top_level_activations)
        top_level_active.scatter_(dim=1, index=topk_indices, src=topk_values)

        reconstruction = self.top_level_decoder(top_level_active)

        active_concepts[:, :self.k] = torch.sort(topk_indices, dim=1).values

        for top_level_idx in torch.unique(topk_indices):
            sub_activations = self.sub_encoders[top_level_idx](x)
            sub_topk_values, sub_topk_indices = torch.topk(sub_activations, k=self.sub_dictionary_k, dim=1)
            
            sub_active = torch.zeros_like(sub_activations)
            sub_active.scatter_(dim=1, index=sub_topk_indices, src=sub_topk_values)

            reconstruction += torch.where(
                torch.any(active_concepts == top_level_idx, dim=1, keepdim=True),
                self.sub_decoders[top_level_idx](sub_active),
                torch.zeros_like(reconstruction)
            )

            adjusted_indices = sub_topk_indices + self.dictionary_size + top_level_idx * self.sub_dictionary_size
            for i in range(self.k):
                active_concepts[:, self.k + self.sub_dictionary_k*i:self.k + self.sub_dictionary_k*(i+1)] = torch.where(
                    active_concepts[:, i:i+1] == top_level_idx,
                    torch.sort(adjusted_indices, dim=1).values,
                    active_concepts[:, self.k + self.sub_dictionary_k*i:self.k + self.sub_dictionary_k*(i+1)]
                )

        return reconstruction, active_concepts

    def run_step(self, batch):
        reconstruction, _ = self.forward(batch[0])

        loss = self.l2_loss(reconstruction, batch[0])

        return loss

    def training_step(self, batch, _):
        loss = self.run_step(batch)
        self.log("loss", float(loss.detach()), prog_bar=True)
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
        return self.forward(batch[0])

    def configure_optimizers(self):
        optimiser = optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser)
        return {
            "optimizer": optimiser,
            "lr_scheduler": lr_scheduler,
            "monitor": "loss",
        }
