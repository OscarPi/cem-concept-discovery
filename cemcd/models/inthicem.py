import numpy as np
import torch
from cemcd.models import hicem
from cemcd.metrics import calculate_concept_accuracies, calculate_task_accuracy

# TODO: not sure about interaction between implied interventions and this

class IntAwareHierarchicalConceptEmbeddingModel(hicem.HierarchicalConceptEmbeddingModel):
    def __init__(
        self,
        concepts,
        n_tasks,
        latent_representation_size,
        embedding_size,
        concept_loss_weight,
        task_class_weights,
        concept_loss_weights,
        intervention_task_discount=1.1,
        intervention_weight=5, # Maybe also try 1
        # Parameters regarding how we select how many concepts to intervene on
        # in the horizon of a current trajectory (this is the length of the
        # trajectory)
        max_horizon=6,
        initial_horizon=2,
        horizon_rate=1.005,
    ):
        super().__init__(
            concepts=concepts,
            n_tasks=n_tasks,
            latent_representation_size=latent_representation_size,
            embedding_size=embedding_size,
            concept_loss_weight=concept_loss_weight,
            task_class_weights=task_class_weights,
            concept_loss_weights=concept_loss_weights,
        )

        units = [
            self.n_top_level_concepts * self.embedding_size + self.n_concepts, # Bottleneck and Prev interventions
            256,
            128,
            self.n_concepts
        ]

        layers = []
        for i in range(1, len(units)):
            layers.append(
                torch.nn.BatchNorm1d(num_features=units[i-1]),
            )
            layers.append(torch.nn.Linear(units[i-1], units[i]))
            if i != len(units) - 1:
                layers.append(torch.nn.LeakyReLU())

        self.concept_rank_model = torch.nn.Sequential(*layers)

        self.intervention_task_discount = intervention_task_discount
        self.horizon_rate = horizon_rate
        self.horizon_limit = torch.nn.Parameter(
            torch.FloatTensor([initial_horizon]),
            requires_grad=False,
        )
        self.current_steps = torch.nn.Parameter(
            torch.IntTensor([0]),
            requires_grad=False,
        )
        self.intervention_weight = intervention_weight
        self.loss_interventions = torch.nn.CrossEntropyLoss()
        self.max_horizon = max_horizon

        self._horizon_distr = lambda init, end: np.random.randint(
            init,
            end,
        )

    # def get_concept_int_distribution(
    #     self,
    #     x,
    #     c,
    #     prev_interventions=None,
    #     competencies=None,
    #     horizon=1,
    # ):
    #     if prev_interventions is None:
    #         prev_interventions = torch.zeros(c.shape).to(x.device)
    #     outputs = self._forward(
    #         x,
    #         c=c,
    #         y=None,
    #         train=False,
    #         competencies=competencies,
    #         prev_interventions=prev_interventions,
    #         output_embeddings=True,
    #         output_latent=True,
    #         output_interventions=True,
    #     )

    #     predicted_concept_probs, c_logits, y_logits = outputs[0], outputs[1], outputs[2]
    #     prev_interventions = outputs[3]
    #     pos_embeddings = outputs[-2]
    #     neg_embeddings = outputs[-1]
    #     return self.prior_int_distribution(
    #         prob=predicted_concept_probs,
    #         pos_embeddings=pos_embeddings,
    #         neg_embeddings=neg_embeddings,
    #         c=c,
    #         competencies=competencies,
    #         prev_interventions=prev_interventions,
    #         horizon=horizon,
    #         train=False,
    #     )

    def prior_int_distribution(
        self,
        concept_probs,
        positive_and_negative_concept_embeddings,
        c,
        prev_interventions=None,
        train=False,
    ):
        if prev_interventions is None:
            prev_interventions = torch.zeros_like(concept_probs)

        # Shape is [B, n_concepts, emb_size]
        concept_probs = prev_interventions * c + (1 - prev_interventions) * concept_probs
        bottleneck = self.mix_embeddings(positive_and_negative_concept_embeddings, concept_probs)

        # Zero out embeddings of previously intervened concepts #TODO: what if train?
        available_concepts = (1 - prev_interventions).to(bottleneck.device)
        used_concepts = 1 - available_concepts
        rank_input = torch.concat((bottleneck, prev_interventions), dim=-1)
        next_concept_scores = self.concept_rank_model(rank_input)

        if train:
            return next_concept_scores

        next_concept_scores = torch.where(
            used_concepts == 1,
            torch.ones_like(used_concepts) * (-1000),
            next_concept_scores,
        )
        return torch.nn.functional.softmax(
            next_concept_scores,
            dim=-1,
        )

    def rollout_y_logits(
        self,
        predicted_concept_probs,
        c_true,
        positive_and_negative_concept_embeddings,
        interventions,
    ):
        concept_probs_after_interventions = (
            predicted_concept_probs * (1 - interventions) +
            c_true * interventions
        )

        # Compute the bottleneck using the mixture of embeddings based
        # on their assigned probabilities
        bottleneck = self.mix_embeddings(
            positive_and_negative_concept_embeddings=positive_and_negative_concept_embeddings,
            concept_probs=concept_probs_after_interventions,
        )
        # Predict the output task logits with the given
        rollout_y_logits = self.label_predictor(bottleneck)

        return rollout_y_logits

    def get_target_mask(
        self,
        y,
        c,
        predicted_concept_probs,
        positive_and_negative_concept_embeddings,
        prev_interventions
    ):
        # Generate as a label the concept which increases the
        # probability of the correct class the most when
        # intervened on
        target_int_logits = torch.ones_like(c) * (-torch.inf)
        for target_concept in range(target_int_logits.shape[-1]):
            new_int = torch.zeros_like(prev_interventions)
            new_int[:, target_concept] = 1

            # Make this intervention and lets see how the y logits change
            updated_int = torch.clamp(
                prev_interventions.detach() + new_int,
                0,
                1,
            )
            updated_int = self.calculate_implied_interventions(updated_int, c)
            rollout_y_logits = self.rollout_y_logits( # TODO: old code was extra, don't need to apply new_int twice.
                interventions=updated_int,
                predicted_concept_probs=predicted_concept_probs.detach(),
                c_true=c,
                positive_and_negative_concept_embeddings=list(map(lambda t: t.detach(), positive_and_negative_concept_embeddings)),
            )

            if self.n_tasks > 1:
                one_hot_y = torch.nn.functional.one_hot(y, self.n_tasks)
                target_int_logits[:, target_concept] = \
                    rollout_y_logits[
                        one_hot_y.type(torch.BoolTensor)
                    ]
            else:
                pred_y_prob = torch.sigmoid(
                    torch.squeeze(rollout_y_logits, dim=-1)
                )
                target_int_logits[:, target_concept] = torch.where(
                    y == 1,
                    torch.log(
                        (pred_y_prob + 1e-15) /
                        (1 - pred_y_prob + 1e-15)
                    ),
                    torch.log(
                        (1 - pred_y_prob + 1e-15) /
                        (pred_y_prob+ 1e-15)
                    ),
                )

        target_int_labels = torch.argmax(target_int_logits, -1)

        return target_int_labels

    def setup_intervention_trajectory(
        self,
        prev_num_of_interventions,
        interventions,
        free_concepts,
    ):
        # The limit of how many concepts we can intervene at most
        int_basis_lim = self.n_concepts
        # And the limit of how many concepts we will intervene at most during
        # this training round
        horizon_lim = int(self.horizon_limit.detach().cpu().numpy()[0])

        # Here we first determine how many concepts we will intervene on at
        # the begining of the trajectory before we even start tallying up
        # losses from this intervention trajectory.
        # We will also sample the length of the trajectory for this training
        # step (current_horizon) as well as the normalization coefficients
        # for the trajectory-dependent losses (task_trajectory_weight and
        # trajectory_weight)
        if prev_num_of_interventions != int_basis_lim: # TODO: not sure what is going on here
            bottom = min(
                horizon_lim,
                int_basis_lim - prev_num_of_interventions - 1,
            )  # -1 so that we at least intervene on one concept
            if bottom > 0:
                initially_selected = np.random.randint(0, bottom)
            else:
                initially_selected = 0

            # Get the maximum size of any current trajectories:
            end_horizon = min(
                int(horizon_lim),
                self.max_horizon,
                int_basis_lim - prev_num_of_interventions - initially_selected,
            )

            # And select the number of steps T we will run the current
            # trajectory for
            current_horizon = self._horizon_distr(
                init=1 if end_horizon > 1 else 0,
                end=end_horizon,
            )

            # At the begining of the trajectory, we start with a total of
            # `initially_selected`` concepts already intervened on. So to
            # indicate that, we will update the intervention_idxs matrix
            # accordingly
            for sample_idx in range(interventions.shape[0]):
                probs = free_concepts[sample_idx, :].detach().cpu().numpy()
                probs = probs/np.sum(probs)
                interventions[
                    sample_idx,
                    np.random.choice(
                        int_basis_lim,
                        size=initially_selected,
                        replace=False,
                        p=probs,
                    )
                ] = 1
            discount = 1
            trajectory_weight = 0
            for i in range(current_horizon):
                trajectory_weight += discount
            task_discount = 1
            task_trajectory_weight = 1
            for i in range(current_horizon):
                task_discount *= self.intervention_task_discount
                if i == current_horizon - 1:
                    task_trajectory_weight += task_discount
            task_discount = 1
        else:
            # Else we will peform no intervention in this training step!
            current_horizon = 0
            task_discount = 1
            task_trajectory_weight = 1
            trajectory_weight = 1

        return (
            current_horizon,
            task_discount,
            task_trajectory_weight,
            trajectory_weight,
            interventions
        )

    def compute_task_loss(
        self,
        y,
        concept_probs=None,
        positive_and_negative_concept_embeddings=None,
        y_pred_logits=None,
    ):
        if y_pred_logits is not None:
            return self.loss_task(y_pred_logits.squeeze(), y)

        bottleneck = self.mix_embeddings(positive_and_negative_concept_embeddings, concept_probs)
        y_logits = self.label_predictor(bottleneck)

        return self.loss_task(y_logits.squeeze(), y)

    def intervention_rollout_loss(
        self,
        y,
        c,
        y_pred_logits,
        predicted_concept_probs,
        positive_and_negative_concept_embeddings,
        prev_intervention_mask,
    ):
        intervention_task_loss = 0.0
        current_horizon = -1
        intervention_loss = 0.0

        interventions = torch.zeros_like(predicted_concept_probs)
        # First, figure out which concepts are free for
        # us to intervene on next
        if prev_intervention_mask is not None:
            # This will be not None in the case of RandInt
            for concept_idx in range(self.n_concepts):
                if prev_intervention_mask[concept_idx] == 1:
                    interventions[:, concept_idx] = True


            free_concepts = 1 - interventions
            prev_num_of_interventions = int(np.sum(
                    prev_intervention_mask.detach().cpu().numpy(),
                ),
            )
        else:
            # Else, we start from a fresh slate without any previous
            # interventions
            prev_num_of_interventions = 0
            free_concepts = torch.ones_like(c)

        # Update the intervention idxs so that we included a small random
        # number of initial interventions (a-la RandInt) at the begining
        # of this trajectory. While we do that, we will also compute the
        # size of the current trajectory T (i.e., the current_horizon) as well
        # as the discounts and weights that come with a trajectory of this
        # size and the initial number of intervened concepts
        (
            current_horizon,
            task_discount,
            task_trajectory_weight,
            trajectory_weight,
            interventions
        ) = self.setup_intervention_trajectory(
            prev_num_of_interventions,
            interventions,
            free_concepts,
        )

        # Then we initialize the intervention trajectory task loss to
        # that of the unintervened model as this loss is not going to
        # be taken into account if we don't do this
        intervention_task_loss = self.compute_task_loss(y=y, y_pred_logits=y_pred_logits)
        intervention_task_loss = intervention_task_loss / task_trajectory_weight

        # And as many steps in the trajectory as suggested
        for i in range(current_horizon):
            # And generate a probability distribution over previously TODO: are you sure this is over previously unseen concepts?
            # unseen concepts to indicate which one we should intervene
            # on next!
            concept_scores = self.prior_int_distribution(
                concept_probs=predicted_concept_probs,
                positive_and_negative_concept_embeddings=positive_and_negative_concept_embeddings,
                c=c,
                prev_interventions=interventions,
                train=True,
            )

            target_int_labels = self.get_target_mask(
                y=y,
                c=c,
                predicted_concept_probs=predicted_concept_probs,
                positive_and_negative_concept_embeddings=positive_and_negative_concept_embeddings,
                prev_interventions=interventions,
            )

            new_loss = self.loss_interventions(concept_scores, target_int_labels)
            # Update the next-concept predictor loss
            intervention_loss += new_loss/trajectory_weight

            # Update the discount (before the task trajectory loss to
            # start discounting from the first intervention so that the
            # loss of the unintervened model is highest
            task_discount *= self.intervention_task_discount

            # Sample the next concepts we will intervene on using a hard
            # Gumbel softmax
            if self.intervention_weight == 0:
                selected_concepts = torch.FloatTensor(
                    np.eye(concept_scores.shape[-1])[np.random.choice(
                        concept_scores.shape[-1],
                        size=concept_scores.shape[0]
                    )]
                ).to(concept_scores.device)
            else:
                selected_concepts = torch.nn.functional.gumbel_softmax(
                    concept_scores,
                    dim=-1,
                    hard=True,
                    tau=1,
                )

            interventions += selected_concepts
            interventions = self.calculate_implied_interventions(interventions, c) # TODO: hmm

            if i == (current_horizon - 1):
                # Then we will also update the task loss with the loss
                # of performing this intervention!
                concept_probs = (
                    predicted_concept_probs * (1 - interventions) +
                    c * interventions
                )
                rollout_y_loss = self.compute_task_loss(
                    y=y,
                    concept_probs=concept_probs,
                    positive_and_negative_concept_embeddings=positive_and_negative_concept_embeddings,
                )
                intervention_task_loss += (
                    task_discount *
                    rollout_y_loss / task_trajectory_weight
                )

        if (self.horizon_limit.detach().cpu().numpy()[0] < self.n_concepts + 1):
            self.horizon_limit *= self.horizon_rate

        intervention_loss = intervention_loss
        intervention_task_loss = intervention_task_loss
        self.current_steps += int(self.intervention_weight > 0)
        return intervention_loss, intervention_task_loss

    def run_step(self, batch, train=False):
        x, y, c = batch

        outputs = self.forward(x, c_true=c, train=train)
        predicted_concept_probs = outputs["predicted_concept_probs"]
        y_logits = outputs["y_logits"]

        # prev_interventions will contain the RandInt intervention mask if
        # we are running this at train time!
        prev_intervention_mask = outputs["intervention_mask"]
        positive_and_negative_concept_embeddings = outputs["positive_and_negative_concept_embeddings"]

        # Then the rollout and imitation learning losses
        if train:
            intervention_loss, intervention_task_loss = self.intervention_rollout_loss(
                c=c,
                predicted_concept_probs=predicted_concept_probs,
                positive_and_negative_concept_embeddings=positive_and_negative_concept_embeddings,
                y=y,
                y_pred_logits=y_logits,
                prev_intervention_mask=prev_intervention_mask,
            )
        else:
            intervention_loss = 0
            intervention_task_loss = self.compute_task_loss(
                y=y,
                concept_probs=predicted_concept_probs,
                positive_and_negative_concept_embeddings=positive_and_negative_concept_embeddings,
            )

        if isinstance(intervention_task_loss, (float, int)):
            intervention_task_loss_scalar = intervention_task_loss
        else:
            intervention_task_loss_scalar = intervention_task_loss.detach()

        if isinstance(intervention_loss, (float, int)):
            intervention_loss_scalar = \
                self.intervention_weight * intervention_loss
        else:
            intervention_loss_scalar = \
                self.intervention_weight * intervention_loss.detach()


        # Finally, compute the concept loss
        concept_loss = self.compute_concept_loss(
            predicted_concept_probs=predicted_concept_probs,
            c=c
        )
        if isinstance(concept_loss, (float, int)):
            concept_loss_scalar = self.concept_loss_weight * concept_loss
        else:
            concept_loss_scalar = \
                self.concept_loss_weight * concept_loss.detach()

        loss = (
            self.concept_loss_weight * concept_loss +
            self.intervention_weight * intervention_loss +
            intervention_task_loss
        )

        # compute accuracy
        if self.concept_loss_weight > 0:
            c_accuracy, c_accuracies, c_auc, c_aucs = calculate_concept_accuracies(predicted_concept_probs, c)
        y_accuracy = calculate_task_accuracy(y_logits, y)

        result = {
            "c_accuracy": c_accuracy,
            "c_auc": c_auc,
            "c_accuracies": c_accuracies,
            "c_aucs": c_aucs,
            "y_accuracy": y_accuracy,
            "concept_loss": concept_loss_scalar,
            "intervention_task_loss": intervention_task_loss_scalar,
            "task_loss": 0, # As the actual task loss is included above!
            "intervention_loss": intervention_loss_scalar,
            "loss": loss.detach() if not isinstance(loss, float) else loss,
            "horizon_limit": self.horizon_limit.detach().cpu().numpy()[0],
        }
        result["current_steps"] = self.current_steps.detach().cpu().numpy()[0]
        return loss, result
