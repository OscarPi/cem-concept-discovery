from cemcd.models.cem import ConceptEmbeddingModel
from cemcd.models.resnet_decoder import get_resnet_decoder
from cemcd.metrics import calculate_concept_accuracies, calculate_task_accuracy
import torch
import numpy as np
import lightning
from torchvision.models import resnet34

class ClusterCEM(ConceptEmbeddingModel):
    def __init__(self,
            n_concepts,
            n_tasks,
            pre_concept_model,
            task_class_weights,
            concept_loss_weights,
            pretrained_concept_embedding_generators=None,
            pretrained_scoring_function=None,
            n_clusters=25):
        
        super().__init__(
            n_concepts=n_concepts,
            n_tasks=n_tasks,
            pre_concept_model=pre_concept_model,
            task_class_weights=task_class_weights,
            concept_loss_weights=concept_loss_weights,
            pretrained_concept_embedding_generators=pretrained_concept_embedding_generators,
            pretrained_scoring_function=pretrained_scoring_function
        )

        self.n_clusters = n_clusters
        self.cluster_loss_weight = 1
        self.cluster_centres = None

        self.reconstruction_loss_weight = 0.1

        self.decoder = get_resnet_decoder(self.embedding_size * self.n_concepts, (299, 299))

        # channels = 3
        # intermediate_maps = 16
        # height = width = 28
        # self.decoder = torch.nn.Sequential(
        #     torch.nn.Linear(
        #         self.embedding_size * self.n_concepts,
        #         width*height*intermediate_maps,
        #     ),
        #     torch.nn.Unflatten(1, (intermediate_maps, height, width)),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.ConvTranspose2d(
        #         in_channels=intermediate_maps,
        #         out_channels=intermediate_maps,
        #         kernel_size=(3,3),
        #         padding=1,
        #     ),
        #     torch.nn.BatchNorm2d(num_features=intermediate_maps),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.ConvTranspose2d(
        #         in_channels=intermediate_maps,
        #         out_channels=intermediate_maps,
        #         kernel_size=(3,3),
        #         padding=1,
        #     ),
        #     torch.nn.BatchNorm2d(num_features=intermediate_maps),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.ConvTranspose2d(
        #         in_channels=intermediate_maps,
        #         out_channels=intermediate_maps,
        #         kernel_size=(3,3),
        #         padding=1,
        #     ),
        #     torch.nn.BatchNorm2d(num_features=intermediate_maps),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.ConvTranspose2d(
        #         in_channels=intermediate_maps,
        #         out_channels=channels,
        #         kernel_size=(3,3),
        #         padding=1,
        #     ),
        #     torch.nn.BatchNorm2d(num_features=channels),
        #     torch.nn.Sigmoid()
        # )

    @torch.no_grad()
    def on_train_start(self):
        embeddings = []
        for i, batch in enumerate(self.trainer.train_dataloader):
            batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
            _, _, batch_embeddings = self.predict_step(batch, i)
            embeddings.append(batch_embeddings)
        embeddings = torch.cat(embeddings)

        self.train_dataset_length = embeddings.shape[0]
        self.cluster_assignments = torch.randint(0, self.n_clusters, (self.n_concepts, self.train_dataset_length))

        self.recalculate_cluster_centres(embeddings)

    @torch.no_grad()
    def recalculate_cluster_centres(self, embeddings):
        split_embeddings = torch.split(embeddings, self.embedding_size, dim=1)

        def calculate_centres(cluster_assignments, embeddings):
            centres = torch.stack([
                embeddings[cluster_assignments == cluster_idx].mean(dim=0)
                for cluster_idx in range(self.n_clusters)
            ])
            return centres

        self.cluster_centres = torch.stack([
            calculate_centres(self.cluster_assignments[i], split_embeddings[i])
            for i in range(self.n_concepts)
        ])

    @torch.no_grad()
    def recalculate_cluster_assignments(self, embeddings):
        split_embeddings = torch.split(embeddings, self.embedding_size, dim=1)

        for i in range(self.n_concepts):
            expanded_centres = self.cluster_centres[i].expand(self.train_dataset_length, -1, -1)
            expanded_embeddings = split_embeddings[i].unsqueeze(1).expand(-1, self.n_clusters, -1)
            distances = torch.linalg.vector_norm(expanded_centres - expanded_embeddings, dim=2)
            distances = torch.nan_to_num(distances, nan=torch.inf)
            self.cluster_assignments[i] = torch.argmin(distances, dim=1)


    def run_step(self, batch, train=False):
        x, y, c = batch

        predicted_concept_probs, predicted_labels, concept_embeddings = self.forward(
            x,
            c_true=c,
            train=train
        )

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

        # Cluster loss
        cluster_loss = 0

        if self.cluster_centres is not None:
            split_embeddings = torch.split(concept_embeddings, self.embedding_size, dim=1)
            for i in range(self.n_concepts):
                valid_clusters = torch.logical_not(torch.isnan(self.cluster_centres[i].sum(dim=1)))
                expanded_centres = self.cluster_centres[i, valid_clusters].expand(concept_embeddings.shape[0], -1, -1)
                expanded_embeddings = split_embeddings[i].unsqueeze(1).expand(-1, valid_clusters.sum(), -1)
                distances = torch.linalg.vector_norm(expanded_centres - expanded_embeddings, dim=2)
                cluster_loss += torch.mean(torch.min(distances, dim=1).values) / self.n_concepts

        # Reconstruction loss
        reconstruction_loss = 0
        if train:
            reconstructed = self.decoder(concept_embeddings)
            reconstruction_loss = torch.nn.functional.mse_loss(x, reconstructed, reduction="none")
            reconstruction_loss = reconstruction_loss.sum(dim=[1, 2, 3]).mean(dim=[0])

        loss = self.concept_loss_weight * concept_loss + task_loss + self.cluster_loss_weight * cluster_loss + self.reconstruction_loss_weight * reconstruction_loss

        y_accuracy = calculate_task_accuracy(predicted_labels, y)

        result = {
            "c_accuracy": c_accuracy,
            "c_accuracies": c_accuracies,
            "c_auc": c_auc,
            "c_aucs": c_aucs,
            "y_accuracy": y_accuracy
        }

        return loss, result
    
    @torch.no_grad()
    def on_train_epoch_end(self):
        embeddings = []
        for i, batch in enumerate(self.trainer.train_dataloader):
            batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)
            _, _, batch_embeddings = self.predict_step(batch, i)
            embeddings.append(batch_embeddings)
        embeddings = torch.cat(embeddings)

        self.recalculate_cluster_centres(embeddings)
        self.recalculate_cluster_assignments(embeddings)
