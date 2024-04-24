import torch
from cemcd.models import base

class ConceptEmbeddingModel(base.BaseModel):
    def __init__(self, n_concepts, n_tasks, pre_concept_model, task_class_weights, concept_loss_weights):
        super().__init__(n_tasks, task_class_weights, concept_loss_weights)
        self.n_concepts = n_concepts
        self.pre_concept_model = pre_concept_model()
        self.embedding_size = 16
        self.concept_loss_weight = 10

        latent_representation_size = list(self.pre_concept_model.modules())[-1].out_features
        self.concept_embedding_generators = torch.nn.ModuleList()
        for _ in range(self.n_concepts):
            self.concept_embedding_generators.append(torch.nn.Sequential(
                torch.nn.Linear(latent_representation_size, self.embedding_size * 2),
                torch.nn.LeakyReLU()
            ))

        self.scoring_function = torch.nn.Linear(self.embedding_size * 2, 1)

        self.label_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.n_concepts * self.embedding_size, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, self.n_tasks)
        )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, c_true=None, train=False):
        latent = self.pre_concept_model(x)

        concept_embeddings = []
        predicted_concept_probs = []

        for generator in self.concept_embedding_generators:
            embedding = generator(latent)
            concept_embeddings.append(embedding)
            predicted_concept_probs.append(self.sigmoid(self.scoring_function(embedding)))

        concept_embeddings = torch.stack(concept_embeddings, dim=1)
        predicted_concept_probs = torch.cat(predicted_concept_probs, axis=-1)

        interventions = None
        if self.intervention_mask is not None:
            interventions = torch.tile(self.intervention_mask, (predicted_concept_probs.shape[0], 1))

        if train and c_true is not None and interventions is None:
            mask = torch.bernoulli(torch.full((self.n_concepts,), 0.25))
            interventions = torch.tile(mask, (predicted_concept_probs.shape[0], 1))

        if c_true is not None and interventions is not None:
            interventions = interventions.to(predicted_concept_probs.device)

            c_true = torch.where(
                torch.logical_or(c_true == 0, c_true == 1),
                c_true,
                predicted_concept_probs
            )

            concept_probs_after_interventions = predicted_concept_probs * (1 - interventions) + interventions * c_true
        else:
            concept_probs_after_interventions = predicted_concept_probs

        mixed_concept_embeddings = (
            concept_embeddings[:, :, :self.embedding_size] * torch.unsqueeze(concept_probs_after_interventions, dim=-1) +
            concept_embeddings[:, :, self.embedding_size:] * (1 - torch.unsqueeze(concept_probs_after_interventions, dim=-1))
        )
        mixed_concept_embeddings = mixed_concept_embeddings.view((-1, self.embedding_size * self.n_concepts))
        predicted_labels = self.label_predictor(mixed_concept_embeddings)

        return predicted_concept_probs, predicted_labels, mixed_concept_embeddings
