import copy
import torch
from cemcd.models import base

class HierarchicalConceptEmbeddingModel(base.BaseModel):
    def __init__(
            self,
            sub_concepts,
            n_tasks,
            pre_concept_model,
            latent_representation_size,
            task_class_weights,
            concept_loss_weights):
        super().__init__(n_tasks, task_class_weights, concept_loss_weights)

        self.n_top_concepts = len(sub_concepts)
        self.sub_concepts = sub_concepts
        self.n_sub_concepts = sum(map(sum, self.sub_concepts))
        self.n_concepts = self.n_top_concepts + self.n_sub_concepts

        self.pre_concept_model = copy.deepcopy(pre_concept_model)

        self.embedding_size = 16
        self.concept_loss_weight = 10

        self.top_concept_embedding_generators = torch.nn.ModuleList()
        for i in range(self.n_top_concepts):
            self.top_concept_embedding_generators.append(torch.nn.Sequential(
                torch.nn.Linear(latent_representation_size, self.embedding_size * 2),
                torch.nn.LeakyReLU()
            ))

        self.sub_concept_embedding_generators = torch.nn.ModuleList()
        for _ in range(self.n_sub_concepts):
            self.sub_concept_embedding_generators.append(torch.nn.Sequential(
                torch.nn.Linear(self.embedding_size, self.embedding_size),
                torch.nn.LeakyReLU()
            ))

        self.scoring_function = torch.nn.Linear(self.embedding_size, 1)

        self.label_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.n_top_concepts * self.embedding_size, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, self.n_tasks)
        )

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)

    def calculate_interventions(self, c_true, intervention_mask):
        device = c_true.device
        batch_size = c_true.shape[0]

        top_concept_interventions = torch.full((batch_size, self.n_top_concepts), torch.nan, device=device)
        sub_concept_interventions = torch.full((batch_size, self.n_sub_concepts), torch.nan, device=device)

        if c_true is not None and intervention_mask is not None:
            for top_concept_idx in range(self.n_top_concepts):
                first_sub_concept_idx = sum(map(sum, self.sub_concepts[:top_concept_idx]))
                first_negative_sub_concept_idx = first_sub_concept_idx + self.sub_concepts[top_concept_idx][0]
                end_of_sub_concepts = first_sub_concept_idx + sum(self.sub_concepts[top_concept_idx])
                for sub_concept_idx in range(first_sub_concept_idx, end_of_sub_concepts):
                    if intervention_mask[self.n_top_concepts + sub_concept_idx] == 1:
                        if sub_concept_idx < first_negative_sub_concept_idx:
                            top_concept_interventions[:, top_concept_idx] = torch.where(
                                c_true[:, self.n_top_concepts + sub_concept_idx] == 1,
                                1,
                                top_concept_interventions[:, top_concept_idx]
                            )
                            sub_concept_interventions[:, first_sub_concept_idx:first_negative_sub_concept_idx] = torch.where(
                                c_true[:, self.n_top_concepts + sub_concept_idx, torch.newaxis] == 1,
                                0,
                                sub_concept_interventions[:, first_sub_concept_idx:first_negative_sub_concept_idx]
                            )
                        else:
                            top_concept_interventions[:, top_concept_idx] = torch.where(
                                c_true[:, self.n_top_concepts + sub_concept_idx] == 1,
                                0,
                                top_concept_interventions[:, top_concept_idx]
                            )
                            sub_concept_interventions[:, first_negative_sub_concept_idx:end_of_sub_concepts] = torch.where(
                                c_true[:, self.n_top_concepts + sub_concept_idx, torch.newaxis] == 1,
                                0,
                                sub_concept_interventions[:, first_negative_sub_concept_idx:end_of_sub_concepts]
                            )
                        sub_concept_interventions[:, sub_concept_idx] = torch.where(
                            torch.logical_or(
                                c_true[:, self.n_top_concepts + sub_concept_idx] == 0,
                                c_true[:, self.n_top_concepts + sub_concept_idx] == 1
                            ),
                            c_true[:, self.n_top_concepts + sub_concept_idx],
                            sub_concept_interventions[:, sub_concept_idx]
                        )

                if intervention_mask[top_concept_idx] == 1:
                    top_concept_interventions[:, top_concept_idx] = torch.where(
                        torch.logical_or(
                            c_true[:, top_concept_idx] == 0,
                            c_true[:, top_concept_idx] == 1
                        ),
                        c_true[:, top_concept_idx],
                        top_concept_interventions[:, top_concept_idx]
                    )

        return top_concept_interventions, sub_concept_interventions

    def run_sub_concept_module(self, top_concept_idx, positive_subconcepts, top_concept_embeddings, sub_concept_interventions):
        batch_size = top_concept_embeddings.shape[0]
        device = top_concept_embeddings.device

        n_sub_concepts = self.sub_concepts[top_concept_idx][0 if positive_subconcepts else 1]
        first_sub_concept_idx = sum(map(sum, self.sub_concepts[:top_concept_idx]))
        if not positive_subconcepts:
            first_sub_concept_idx += self.sub_concepts[top_concept_idx][0]

        predicted_sub_concept_probs = torch.zeros((batch_size, 0), device=device)
        embeddings = torch.zeros((batch_size, 0, self.embedding_size), device=device)

        for sub_concept_idx in range(first_sub_concept_idx, first_sub_concept_idx + n_sub_concepts):
            sub_concept_embeddings = self.sub_concept_embedding_generators[sub_concept_idx](top_concept_embeddings)
            sub_concept_probs = self.sigmoid(self.scoring_function(sub_concept_embeddings))
            predicted_sub_concept_probs = torch.cat((predicted_sub_concept_probs, sub_concept_probs), dim=1)
            embeddings = torch.cat((embeddings, sub_concept_embeddings[:, torch.newaxis, :]), dim=1)

        sub_concept_probs_after_interventions = torch.where(
            torch.logical_not(torch.isnan(sub_concept_interventions[:, first_sub_concept_idx:first_sub_concept_idx+n_sub_concepts])),
            sub_concept_interventions[:, first_sub_concept_idx:first_sub_concept_idx+n_sub_concepts],
            predicted_sub_concept_probs
        )

        embeddings = torch.sum(embeddings * sub_concept_probs_after_interventions[:, :, torch.newaxis], dim=1)

        return predicted_sub_concept_probs, sub_concept_probs_after_interventions, embeddings

    def forward(self, x, c_true=None, train=False):
        batch_size = x.shape[0]
        device = x.device
        if self.pre_concept_model is None:
            latent = x
        else:
            latent = self.pre_concept_model(x)

        intervention_mask = self.intervention_mask
        if train and c_true is not None and intervention_mask is None:
            intervention_mask = torch.bernoulli(torch.full((self.n_concepts,), 0.25))
        top_concept_interventions, sub_concept_interventions = self.calculate_interventions(c_true, intervention_mask)

        all_mixed_embeddings = torch.zeros((batch_size, 0), device=device)
        all_predicted_top_concept_probs = torch.zeros((batch_size, 0), device=device)
        all_predicted_sub_concept_probs = torch.zeros((batch_size, 0), device=device)

        for top_concept_idx, top_concept_generator in enumerate(self.top_concept_embedding_generators):
            top_concept_embeddings = top_concept_generator(latent)
            top_concept_positive_embeddings = top_concept_embeddings[:, :self.embedding_size]
            top_concept_negative_embeddings = top_concept_embeddings[:, self.embedding_size:]

            if self.sub_concepts[top_concept_idx][0] > 0:
                predicted_sub_concept_probs, sub_concept_probs_after_interventions, top_positive_embeddings = self.run_sub_concept_module(
                    top_concept_idx=top_concept_idx,
                    positive_subconcepts=True,
                    top_concept_embeddings=top_concept_positive_embeddings,
                    sub_concept_interventions=sub_concept_interventions
                )
                all_predicted_sub_concept_probs = torch.cat((all_predicted_sub_concept_probs, predicted_sub_concept_probs), dim=1)
                predicted_top_concept_probs_1 = torch.sum(self.softmax(200*predicted_sub_concept_probs - 100) * predicted_sub_concept_probs, dim=1)
                top_concept_probs_1_after_sub_concept_interventions = torch.sum(self.softmax(200*sub_concept_probs_after_interventions - 100) * sub_concept_probs_after_interventions, dim=1)
            else:
                top_positive_embeddings = top_concept_positive_embeddings
                predicted_top_concept_probs_1 = self.sigmoid(self.scoring_function(top_positive_embeddings)).squeeze()
                top_concept_probs_1_after_sub_concept_interventions = predicted_top_concept_probs_1
            if self.sub_concepts[top_concept_idx][1] > 0:
                predicted_sub_concept_probs, sub_concept_probs_after_interventions, top_negative_embeddings = self.run_sub_concept_module(
                    top_concept_idx=top_concept_idx,
                    positive_subconcepts=False,
                    top_concept_embeddings=top_concept_negative_embeddings,
                    sub_concept_interventions=sub_concept_interventions
                )
                all_predicted_sub_concept_probs = torch.cat((all_predicted_sub_concept_probs, predicted_sub_concept_probs), dim=1)
                predicted_top_concept_probs_2 = 1 - torch.sum(self.softmax(200*predicted_sub_concept_probs - 100) * predicted_sub_concept_probs, dim=1)
                top_concept_probs_2_after_sub_concept_interventions = 1 - torch.sum(self.softmax(200*sub_concept_probs_after_interventions - 100) * sub_concept_probs_after_interventions, dim=1)
            else:
                top_negative_embeddings = top_concept_negative_embeddings
                predicted_top_concept_probs_2 = 1 - self.sigmoid(self.scoring_function(top_negative_embeddings)).squeeze()
                top_concept_probs_2_after_sub_concept_interventions = predicted_top_concept_probs_2

            predicted_top_concept_probs = (predicted_top_concept_probs_1 + predicted_top_concept_probs_2) / 2
            all_predicted_top_concept_probs = torch.cat((all_predicted_top_concept_probs, predicted_top_concept_probs[:, torch.newaxis]), dim=1)
            top_concept_probs_after_interventions = torch.where(
                torch.logical_not(torch.isnan(top_concept_interventions[:, top_concept_idx])),
                top_concept_interventions[:, top_concept_idx],
                (top_concept_probs_1_after_sub_concept_interventions + top_concept_probs_2_after_sub_concept_interventions) / 2
            )

            all_mixed_embeddings = torch.cat((all_mixed_embeddings, (
                top_positive_embeddings * top_concept_probs_after_interventions[:, torch.newaxis] +
                top_negative_embeddings * (1 - top_concept_probs_after_interventions[:, torch.newaxis])
            )), dim=1)

        predicted_concept_probs = torch.cat((all_predicted_top_concept_probs, all_predicted_sub_concept_probs), axis=-1)

        predicted_labels = self.label_predictor(all_mixed_embeddings)

        return predicted_concept_probs, predicted_labels, all_mixed_embeddings


# class HierarchicalConceptEmbeddingModel(base.BaseModel):
#     def __init__(
#             self,
#             sub_concepts,
#             n_tasks,
#             pre_concept_model,
#             latent_representation_size,
#             task_class_weights,
#             concept_loss_weights):
#         super().__init__(n_tasks, task_class_weights, concept_loss_weights)

#         self.n_top_concepts = len(sub_concepts)
#         self.sub_concepts = sub_concepts
#         self.n_sub_concepts = sum(map(sum, self.sub_concepts))
#         self.n_concepts = self.n_top_concepts + self.n_sub_concepts


#         self.pre_concept_model = copy.deepcopy(pre_concept_model)

#         self.embedding_size = 16
#         self.concept_loss_weight = 10

#         self.concept_embedding_generators = torch.nn.ModuleList()
#         for ns in self.sub_concepts:
#             for n in ns:
#                 for _ in range(max(1, n)):
#                     self.concept_embedding_generators.append(torch.nn.Sequential(
#                         torch.nn.Linear(latent_representation_size, self.embedding_size),
#                         torch.nn.LeakyReLU()
#                     ))

#         self.scoring_function = torch.nn.Linear(self.embedding_size, 1)

#         self.label_predictor = torch.nn.Sequential(
#             torch.nn.Linear(self.n_top_concepts * self.embedding_size, 128),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(128, 128),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(128, self.n_tasks)
#         )

#         self.softmax = torch.nn.Softmax(dim=1)

#     def forward(self, x, c_true=None, train=False):
#         batch_size = x.shape[0]
#         if self.pre_concept_model is None:
#             latent = x
#         else:
#             latent = self.pre_concept_model(x)

#         intervention_mask = self.intervention_mask
#         if train and c_true is not None and intervention_mask is None:
#             intervention_mask = torch.bernoulli(torch.full((self.n_concepts,), 0.25))

#         embeddings = torch.zeros((batch_size, 0, self.embedding_size), device=latent.device)
#         scores = torch.zeros((batch_size, 0), device=latent.device)
#         for generator in self.concept_embedding_generators:
#             embedding = generator(latent)
#             embeddings = torch.cat((embeddings, embedding[:, torch.newaxis, :]), dim=1)
#             score = self.scoring_function(embedding)
#             scores = torch.cat((scores, score), dim=1)

#         first_sub_embedding_idx = 0
#         next_sub_concept_idx = self.n_top_concepts
#         mixed_embeddings = torch.zeros((batch_size, 0), device=latent.device)
#         predicted_top_concept_probs = torch.zeros((batch_size, 0), device=latent.device)
#         predicted_sub_concept_probs = torch.zeros((batch_size, 0), device=latent.device)
#         for top_concept_idx in range(self.n_top_concepts):
#             n_positive_sub_concepts = self.sub_concepts[top_concept_idx][0]
#             n_negative_sub_concepts = self.sub_concepts[top_concept_idx][1]
#             total_n_sub_embeddings = max(1, n_positive_sub_concepts) + max(1, n_negative_sub_concepts)

#             probs = self.softmax(scores[:, first_sub_embedding_idx:first_sub_embedding_idx+total_n_sub_embeddings])
#             probs = probs / (torch.sum(probs, dim=1, keepdim=True) + 0.001)
#             predicted_top_concept_probs = torch.cat((
#                 predicted_top_concept_probs,
#                 torch.sum(probs[:, :max(1, n_positive_sub_concepts)], dim=1, keepdim=True)
#             ), dim=1)

#             if n_positive_sub_concepts > 0 or n_negative_sub_concepts > 0:
#                 if n_positive_sub_concepts == 0:
#                     probs = probs[:, 1:]
#                 if n_negative_sub_concepts == 0:
#                     probs = probs[:, :-1]

#                 predicted_sub_concept_probs = torch.cat((predicted_sub_concept_probs, probs), dim=1)

#             scores_after_interventions = scores[:, first_sub_embedding_idx:first_sub_embedding_idx+total_n_sub_embeddings]

#             if c_true is not None and intervention_mask is not None:
#                 sub_embeddings_to_intervene = range(total_n_sub_embeddings)
#                 if n_positive_sub_concepts == 0:
#                     sub_embeddings_to_intervene = sub_embeddings_to_intervene[1:]
#                 if n_negative_sub_concepts == 0:
#                     sub_embeddings_to_intervene = sub_embeddings_to_intervene[:-1]
#                 for sub_embedding_idx in sub_embeddings_to_intervene:
#                     if intervention_mask[next_sub_concept_idx] == 1:
#                         scores_after_interventions = torch.where(
#                             c_true[:, next_sub_concept_idx, torch.newaxis] == 1,
#                             -torch.inf,
#                             scores_after_interventions
#                         )
#                         scores_after_interventions[:, sub_embedding_idx] = torch.where(
#                             torch.logical_or(
#                                 c_true[:, next_sub_concept_idx] == 0,
#                                 c_true[:, next_sub_concept_idx] == 1
#                             ),
#                             torch.where(c_true[:, next_sub_concept_idx] == 1, 1, -torch.inf),
#                             scores_after_interventions[:, sub_embedding_idx]
#                         )
#                     next_sub_concept_idx += 1
#                 if intervention_mask[top_concept_idx] == 1:
#                     scores_after_interventions = torch.where(
#                         c_true[:, top_concept_idx, torch.newaxis] == 1,
#                         torch.cat((
#                             scores_after_interventions[:, :max(1, n_positive_sub_concepts)],
#                             torch.full((batch_size, max(1, n_negative_sub_concepts)), -torch.inf, device=latent.device)
#                         ), dim=1),
#                         scores_after_interventions
#                     )
#                     scores_after_interventions = torch.where(
#                         c_true[:, top_concept_idx, torch.newaxis] == 0,
#                         torch.cat((
#                             torch.full((batch_size, max(1, n_positive_sub_concepts)), -torch.inf, device=latent.device),
#                             scores_after_interventions[:, max(1, n_positive_sub_concepts):]
#                         ), dim=1),
#                         scores_after_interventions
#                     )

#             scores_after_interventions = torch.where(
#                 torch.all(torch.isinf(scores_after_interventions), dim=1, keepdim=True),
#                 0,
#                 scores_after_interventions
#             )
#             probs_after_interventions = self.softmax(scores_after_interventions)
#             mixed_embeddings = torch.cat((
#                 mixed_embeddings,
#                 torch.sum(
#                     embeddings[:, first_sub_embedding_idx:first_sub_embedding_idx+total_n_sub_embeddings] * probs_after_interventions[:, :, torch.newaxis],
#                     dim=1
#                 )
#             ), dim=1)

#             first_sub_embedding_idx += total_n_sub_embeddings

#         predicted_concept_probs = torch.cat((predicted_top_concept_probs, predicted_sub_concept_probs), axis=-1)

#         predicted_labels = self.label_predictor(mixed_embeddings)

#         return predicted_concept_probs, predicted_labels, mixed_embeddings
