import copy
import torch
from cemcd.models import base

class ConceptBottleneckModel(base.BaseModel):
    def __init__(self, n_concepts, n_tasks, concept_model, task_class_weights, concept_loss_weights, black_box=False):
        super().__init__(n_tasks, task_class_weights, concept_loss_weights)
        self.n_concepts = n_concepts
        self.concept_model = copy.deepcopy(concept_model)
        self.concept_loss_weight = 10
        self.black_box = black_box
        if black_box:
            self.concept_loss_weight = 0

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

            c_true = torch.where(
                torch.logical_or(c_true == 0, c_true == 1),
                c_true,
                predicted_concept_probs
            )

            concept_probs_after_interventions = predicted_concept_probs * (1 - interventions) + interventions * c_true
        else:
            concept_probs_after_interventions = predicted_concept_probs

        if self.black_box:
            predicted_labels = self.label_predictor(predicted_concept_logits)
        else:
            predicted_labels = self.label_predictor(concept_probs_after_interventions)

        return predicted_concept_probs, predicted_labels
