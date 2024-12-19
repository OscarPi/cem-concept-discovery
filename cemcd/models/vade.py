import lightning
import torch
import numpy as np

class VaDE(lightning.LightningModule):
    def __init__(self, n_clusters):
        super().__init__()

        self.n_clusters = n_clusters

        intermediate_maps = 16
        channels = 1
        width = 28
        height = 28
        output_dim = 512
        latent_dim = 64

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channels,
                out_channels=intermediate_maps,
                kernel_size=(3,3),
                padding='same',
            ),
            torch.nn.BatchNorm2d(num_features=intermediate_maps),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=intermediate_maps,
                out_channels=intermediate_maps,
                kernel_size=(3,3),
                padding='same',
            ),
            torch.nn.BatchNorm2d(num_features=intermediate_maps),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=intermediate_maps,
                out_channels=intermediate_maps,
                kernel_size=(3,3),
                padding='same',
            ),
            torch.nn.BatchNorm2d(num_features=intermediate_maps),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=intermediate_maps,
                out_channels=intermediate_maps,
                kernel_size=(3,3),
                padding='same',
            ),
            torch.nn.BatchNorm2d(num_features=intermediate_maps),
            torch.nn.LeakyReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(
                width*height*intermediate_maps,
                output_dim,
            ),
            torch.nn.LeakyReLU()
        )

        self.mu = torch.nn.Linear(output_dim, latent_dim)
        self.log_var = torch.nn.Linear(output_dim, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, width * height * intermediate_maps),
            torch.nn.Unflatten(1, (intermediate_maps, height, width)),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=intermediate_maps,
                out_channels=intermediate_maps,
                kernel_size=(3,3),
                padding='same',
            ),
            torch.nn.BatchNorm2d(num_features=intermediate_maps),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=intermediate_maps,
                out_channels=intermediate_maps,
                kernel_size=(3,3),
                padding='same',
            ),
            torch.nn.BatchNorm2d(num_features=intermediate_maps),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=intermediate_maps,
                out_channels=intermediate_maps,
                kernel_size=(3,3),
                padding='same',
            ),
            torch.nn.BatchNorm2d(num_features=intermediate_maps),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=intermediate_maps,
                out_channels=channels,
                kernel_size=(3,3),
                padding='same',
            ),
            torch.nn.BatchNorm2d(num_features=channels),
            torch.nn.Sigmoid(),
        )

        self.pi = torch.nn.Parameter(torch.ones(self.n_clusters) / self.n_clusters, requires_grad=True)
        self.mu_clusters = torch.nn.Parameter(torch.zeros((self.n_clusters, latent_dim)), requires_grad=True)
        self.log_var_clusters = torch.nn.Parameter(torch.zeros((self.n_clusters, latent_dim)), requires_grad=True)


    def elbo_loss(self, x):
        det = 1e-10

        L_rec=0

        latent = self.encoder(x)
        z_mu, z_sigma2_log = self.encoder(x)

            z=torch.randn_like(z_mu)*torch.exp(z_sigma2_log/2)+z_mu

            x_pro=self.decoder(z)

            L_rec+=F.binary_cross_entropy(x_pro,x)

        Loss=L_rec*x.size(1)

        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c

        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(z,mu_c,log_sigma2_c))+det

        yita_c=yita_c/(yita_c.sum(1).view(-1,1))#batch_size*Clusters

        Loss+=0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))

        Loss-=torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))


        return Loss

    def forward(self, x):
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
            "log": {**result, "loss":float(loss)}
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
