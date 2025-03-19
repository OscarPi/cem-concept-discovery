import copy
import torch
from cemcd.models import base

class HierarchicalConceptEmbeddingModel(base.BaseModel):
    def __init__(
            self,
            n_top_concepts,
            n_sub_concepts,
            n_tasks,
            pre_concept_model,
            latent_representation_size,
            task_class_weights,
            concept_loss_weights):
        super().__init__(n_tasks, task_class_weights, concept_loss_weights)
        self.n_top_concepts = n_top_concepts
        if n_sub_concepts is None:
            n_sub_concepts = [0] * n_top_concepts
        self.n_sub_concepts = n_sub_concepts
        self.n_concepts = self.n_top_concepts + sum(self.n_sub_concepts)

        self.pre_concept_model = copy.deepcopy(pre_concept_model)

        self.embedding_size = 16
        self.concept_loss_weight = 10

        self.top_concept_embedding_generators = torch.nn.ModuleList()
        self.sub_concept_embedding_generators = torch.nn.ModuleList()
        self.embedding_compressors = torch.nn.ModuleList()
        for i in range(self.n_top_concepts):
            self.top_concept_embedding_generators.append(torch.nn.Sequential(
                torch.nn.Linear(latent_representation_size, self.embedding_size * 2),
                torch.nn.LeakyReLU()
            ))
            for _ in range(self.n_sub_concepts[i]):
                self.sub_concept_embedding_generators.append(torch.nn.Sequential(
                    torch.nn.Linear(self.embedding_size, self.embedding_size * 2),
                    torch.nn.LeakyReLU()
                ))
            if self.n_sub_concepts[i] == 0:
                self.embedding_compressors.append(torch.nn.Identity())
            else:
                self.embedding_compressors.append(torch.nn.Sequential(
                    torch.nn.Linear(self.embedding_size * self.n_sub_concepts[i], self.embedding_size),
                    torch.nn.LeakyReLU()
                ))

        self.scoring_function = torch.nn.Linear(self.embedding_size * 2, 1)

        self.label_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.n_top_concepts * self.embedding_size, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, self.n_tasks)
        )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, c_true=None, train=False):
        if self.pre_concept_model is None:
            latent = x
        else:
            latent = self.pre_concept_model(x)

        intervention_mask = self.intervention_mask
        if train and c_true is not None and intervention_mask is None:
            intervention_mask = torch.bernoulli(torch.full((self.n_concepts,), 0.25))

        all_mixed_embeddings = []
        all_predicted_top_concept_probs = []
        all_predicted_sub_concept_probs = []

        first_sub_concept_idx = 0
        for top_concept_idx, top_concept_generator in enumerate(self.top_concept_embedding_generators):
            top_concept_embeddings = top_concept_generator(latent)
            predicted_top_concept_probs = self.sigmoid(self.scoring_function(top_concept_embeddings)).squeeze()
            all_predicted_top_concept_probs.append(predicted_top_concept_probs)

            top_concept_positive_embeddings = top_concept_embeddings[:, :self.embedding_size]
            top_concept_negative_embeddings = top_concept_embeddings[:, self.embedding_size:]

            all_sub_concept_embeddings = []
            for sub_concept_idx in range(first_sub_concept_idx, first_sub_concept_idx + self.n_sub_concepts[top_concept_idx]):
                sub_concept_embeddings = self.sub_concept_embedding_generators[sub_concept_idx](top_concept_positive_embeddings)
                predicted_sub_concept_probs = self.sigmoid(self.scoring_function(sub_concept_embeddings)).squeeze()
                all_predicted_sub_concept_probs.append(predicted_sub_concept_probs)
                sub_concept_probs_after_interventions = predicted_sub_concept_probs
                if c_true is not None and intervention_mask is not None and intervention_mask[self.n_top_concepts + sub_concept_idx] == 1:
                    sub_concept_probs_after_interventions = torch.where(
                        torch.logical_or(
                            c_true[:, self.n_top_concepts + sub_concept_idx] == 0,
                            c_true[:, self.n_top_concepts + sub_concept_idx] == 1
                        ),
                        c_true[:, self.n_top_concepts + sub_concept_idx],
                        predicted_sub_concept_probs
                    )
                    
                mixed_sub_concept_embeddings = (
                    sub_concept_embeddings[:, :self.embedding_size] * torch.unsqueeze(sub_concept_probs_after_interventions, dim=-1) +
                    sub_concept_embeddings[:, self.embedding_size:] * (1 - torch.unsqueeze(sub_concept_probs_after_interventions, dim=-1))
                )
                all_sub_concept_embeddings.append(mixed_sub_concept_embeddings)

            top_positive_embeddings = top_concept_positive_embeddings
            if len(all_sub_concept_embeddings) > 0:
                top_positive_embeddings = self.embedding_compressors[top_concept_idx](torch.cat(all_sub_concept_embeddings, dim=-1))

            top_concept_probs_after_interventions = predicted_top_concept_probs
            if c_true is not None and intervention_mask is not None and intervention_mask[top_concept_idx] == 1:
                top_concept_probs_after_interventions = torch.where(
                    torch.logical_or(
                        c_true[:, top_concept_idx] == 0,
                        c_true[:, top_concept_idx] == 1),
                    c_true[:, top_concept_idx],
                    predicted_top_concept_probs
                )

            all_mixed_embeddings.append(
                top_positive_embeddings * torch.unsqueeze(top_concept_probs_after_interventions, dim=-1) +
                top_concept_negative_embeddings * (1 - torch.unsqueeze(top_concept_probs_after_interventions, dim=-1))
            )

            first_sub_concept_idx += self.n_sub_concepts[top_concept_idx]

        all_mixed_embeddings = torch.cat(all_mixed_embeddings, axis=-1)
        all_predicted_top_concept_probs = torch.stack(all_predicted_top_concept_probs, dim=-1)
        all_predicted_sub_concept_probs = torch.stack(all_predicted_sub_concept_probs, dim=-1)
        predicted_concept_probs = torch.cat((all_predicted_top_concept_probs, all_predicted_sub_concept_probs), axis=-1)

        predicted_labels = self.label_predictor(all_mixed_embeddings)

        return predicted_concept_probs, predicted_labels, all_mixed_embeddings
