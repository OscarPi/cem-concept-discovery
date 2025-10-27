import torch
from cemcd.models import base

class ConceptBottleneckModel(base.BaseModel):
    def __init__(
            self,
            n_concepts,
            n_tasks,
            latent_representation_size,
            concept_loss_weight,
            task_class_weights,
            concept_loss_weights):
        super().__init__(n_tasks, task_class_weights, concept_loss_weights)
        self.n_concepts = n_concepts
        self.concept_loss_weight = concept_loss_weight

        # Representations from the foundation model are precomputed and passed in the dataset.
        self.concept_model = torch.nn.Linear(latent_representation_size, self.n_concepts)

        self.label_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.n_concepts, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, self.n_tasks)
        )

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, c_true=None, train=False):
        predicted_concept_logits = self.concept_model(x)
        predicted_concept_probs = self.sigmoid(predicted_concept_logits)

        interventions = None
        if self.intervention_mask is not None:
            interventions = torch.tile(self.intervention_mask, (predicted_concept_probs.shape[0], 1))

        if c_true is not None and interventions is not None:
            interventions = interventions.to(predicted_concept_probs.device)
            if isinstance(self.intervention_off_value, torch.Tensor):
                intervention_off_value = self.intervention_off_value.to(
                    dtype=torch.float32,
                    device=predicted_concept_probs.device)
            else:
                intervention_off_value = self.intervention_off_value
            if isinstance(self.intervention_on_value, torch.Tensor):
                intervention_on_value = self.intervention_on_value.to(
                    dtype=torch.float32,
                    device=predicted_concept_probs.device)
            else:
                intervention_on_value = self.intervention_on_value

            c_true = torch.where(
                torch.logical_or(c_true == 0, c_true == 1),
                torch.where(c_true == 0, intervention_off_value, intervention_on_value),
                predicted_concept_probs
            )

            concept_probs_after_interventions = predicted_concept_probs * (1 - interventions) + interventions * c_true
        else:
            concept_probs_after_interventions = predicted_concept_probs

        predicted_labels = self.label_predictor(concept_probs_after_interventions)

        return {
            "predicted_concept_probs": predicted_concept_probs,
            "predicted_labels": predicted_labels
        }
