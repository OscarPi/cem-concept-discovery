import torch
from cemcd.models import base

concepts = [
    { # Concept 0
        "name": "...",
        "idx": 0, # Global index of concept in flattened list
        "positive_sub_concepts": [], # List of positive concepts
        "negative_sub_concepts": []  # List of negative sub-concepts
    },
    {}, # Concept 1
    {}, # Concept 2
]
# Construct global concept name as root/parent/.../concept_name

test_concepts = [
    {
        "name": "Fox",
        "idx": 0,
        "positive_sub_concepts": [
            {
                "name": "Red Hair",
                "idx": 1,
                "positive_sub_concepts": [],
                "negative_sub_concepts": []
            },
            {
                "name": "Grey Hair",
                "idx": 2,
                "positive_sub_concepts": [
                    {
                        "name": "Green eyes",
                        "idx": 3,
                        "positive_sub_concepts": [],
                        "negative_sub_concepts": []
                    }
                ],
                "negative_sub_concepts": [
                    {
                        "name": "Blue eyes",
                        "idx": 4,
                        "positive_sub_concepts": [],
                        "negative_sub_concepts": []
                    },
                    {
                        "name": "Black eyes",
                        "idx": 5,
                        "positive_sub_concepts": [],
                        "negative_sub_concepts": []
                    }
                ]
            }
        ],
        "negative_sub_concepts": []
    },
    {
        "name": "Dog",
        "idx": 6,
        "positive_sub_concepts": [
            {
                "name": "Has Tail",
                "idx": 7,
                "positive_sub_concepts": [],
                "negative_sub_concepts": []
            },
            {
                "name": "Is Loyal",
                "idx": 8,
                "positive_sub_concepts": [],
                "negative_sub_concepts": []
            }
        ],
        "negative_sub_concepts": [
            {
                "name": "Is Cat",
                "idx": 9,
                "positive_sub_concepts": [],
                "negative_sub_concepts": []
            }
        ]
    }
]

def count_concepts(concepts):
    def count(concept):
        total = 1
        for sub_concept in concept["positive_sub_concepts"] + concept["negative_sub_concepts"]:
            total += count(sub_concept)
        return total

    total_count = 0
    for concept in concepts:
        total_count += count(concept)
    return total_count

def get_concepts_parents(concepts):
    positive_parents = [None] * count_concepts(concepts)
    negative_parents = [None] * count_concepts(concepts)

    def find_parents(current_positive_parents, current_negative_parents, concept):
        positive_parents[concept["idx"]] = current_positive_parents
        negative_parents[concept["idx"]] = current_negative_parents

        for sub_concept in concept["positive_sub_concepts"]:
            find_parents(
                current_positive_parents + [concept["idx"]],
                current_negative_parents,
                sub_concept
            )
        for sub_concept in concept["negative_sub_concepts"]:
            find_parents(
                current_positive_parents,
                current_negative_parents + [concept["idx"]],
                sub_concept
            )

    for concept in concepts:
        find_parents([], [], concept)

    return positive_parents, negative_parents

def get_concept_names(concepts):
    concept_names = [None] * count_concepts(concepts)

    def find_names(current_name_prefix, concept):
        concept_names[concept["idx"]] = current_name_prefix + concept["name"]

        for sub_concept in concept["positive_sub_concepts"]:
            find_names(
                current_name_prefix + concept["name"] + "/",
                sub_concept
            )
        for sub_concept in concept["negative_sub_concepts"]:
            find_names(
                current_name_prefix + concept["name"] + "\\",
                sub_concept
            )

    for concept in concepts:
        find_names("", concept)

    return concept_names

def get_flat_concept_list(concepts):
    flat_list = [None] * count_concepts(concepts)

    def flatten(concept):
        flat_list[concept["idx"]] = concept
        for sub_concept in concept["positive_sub_concepts"] + concept["negative_sub_concepts"]:
            flatten(sub_concept)

    for concept in concepts:
        flatten(concept)

    return flat_list

class HierarchicalConceptEmbeddingModel(base.BaseModel):
    def __init__(
            self,
            concepts,
            n_tasks,
            latent_representation_size,
            embedding_size,
            concept_loss_weight,
            task_class_weights,
            concept_loss_weights):
        super().__init__(n_tasks, task_class_weights, concept_loss_weights)

        self.concepts = concepts
        self.n_concepts = count_concepts(self.concepts)
        self.n_top_level_concepts = len(self.concepts)
        self.positive_parents, self.negative_parents = get_concepts_parents(self.concepts)

        self.concept_names = get_concept_names(self.concepts)

        self.embedding_size = embedding_size
        self.concept_loss_weight = concept_loss_weight

        self.embedding_generators = torch.nn.ModuleList()
        self.positive_embedding_compressors = torch.nn.ModuleList()
        self.negative_embedding_compressors = torch.nn.ModuleList()
        flat_concept_list = get_flat_concept_list(self.concepts)
        for i in range(self.n_concepts):
            if len(self.positive_parents[i]) == 0 and len(self.negative_parents[i]) == 0:
                self.embedding_generators.append(torch.nn.Sequential(
                    torch.nn.Linear(latent_representation_size, self.embedding_size * 2),
                    torch.nn.LeakyReLU()
                ))
            else:
                self.embedding_generators.append(torch.nn.Sequential(
                    torch.nn.Linear(self.embedding_size, self.embedding_size * 2),
                    torch.nn.LeakyReLU()
                ))
            if len(flat_concept_list[i]["positive_sub_concepts"]) > 1:
                self.positive_embedding_compressors.append(torch.nn.Sequential(
                    torch.nn.Linear(len(flat_concept_list[i]["positive_sub_concepts"]) * self.embedding_size, self.embedding_size),
                    torch.nn.LeakyReLU()
                ))
            else:
                self.positive_embedding_compressors.append(None)
            if len(flat_concept_list[i]["negative_sub_concepts"]) > 1:
                self.negative_embedding_compressors.append(torch.nn.Sequential(
                    torch.nn.Linear(len(flat_concept_list[i]["negative_sub_concepts"]) * self.embedding_size, self.embedding_size),
                    torch.nn.LeakyReLU()
                ))
            else:
                self.negative_embedding_compressors.append(None)

        self.scoring_function = torch.nn.Linear(self.embedding_size * 2, 1)

        self.label_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.n_top_level_concepts * self.embedding_size, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, self.n_tasks)
        )

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)

    def compute_concept_embeddings_and_probs(self, x):
        positive_and_negative_concept_embeddings = [None] * self.n_concepts
        predicted_concept_probs = [None] * self.n_concepts

        def compute_embeddings_and_probs(concept, input_embedding):
            embedding = self.embedding_generators[concept["idx"]](input_embedding)
            positive_and_negative_concept_embeddings[concept["idx"]] = embedding
            predicted_concept_probs[concept["idx"]] = self.sigmoid(self.scoring_function(embedding))

            for sub_concept in concept["positive_sub_concepts"]:
                compute_embeddings_and_probs(sub_concept, embedding[:, :self.embedding_size])
            for sub_concept in concept["negative_sub_concepts"]:
                compute_embeddings_and_probs(sub_concept, embedding[:, self.embedding_size:])

        for concept in self.concepts:
            compute_embeddings_and_probs(concept, x)

        predicted_concept_probs = torch.cat(predicted_concept_probs, axis=-1) # (batch_size, n_concepts)

        return positive_and_negative_concept_embeddings, predicted_concept_probs

    def calculate_implied_interventions(self, interventions, c_true):
        for concept_idx in range(self.n_concepts):
            for parent_idx in self.positive_parents[concept_idx] + self.negative_parents[concept_idx]:
                mask = torch.zeros_like(interventions)
                mask[:, parent_idx] = torch.logical_and(
                    torch.logical_and(interventions[:, concept_idx] == 1, interventions[:, parent_idx] == 0),
                    c_true[:, concept_idx] == 1
                )
                interventions = torch.where(
                    mask,
                    interventions[:, concept_idx],
                    interventions
                )

        return interventions

    def mix_embeddings(self, positive_and_negative_concept_embeddings, concept_probs):
        def mix_embeddings_recursive(concept):
            positive_embeddings = positive_and_negative_concept_embeddings[concept["idx"]][:, :self.embedding_size]
            if len(concept["positive_sub_concepts"]) > 0:
                positive_sub_concept_embeddings = []
                for sub_concept in concept["positive_sub_concepts"]:
                    positive_sub_concept_embeddings.append(mix_embeddings_recursive(sub_concept))
                positive_sub_concept_embeddings = torch.concatenate(positive_sub_concept_embeddings, dim=1)
                positive_embeddings = self.positive_embedding_compressors[concept["idx"]](positive_sub_concept_embeddings)
            
            negative_embeddings = positive_and_negative_concept_embeddings[concept["idx"]][:, self.embedding_size:]
            if len(concept["negative_sub_concepts"]) > 0:
                negative_sub_concept_embeddings = []
                for sub_concept in concept["negative_sub_concepts"]:
                    negative_sub_concept_embeddings.append(mix_embeddings_recursive(sub_concept))
                negative_sub_concept_embeddings = torch.concatenate(negative_sub_concept_embeddings, dim=1)
                negative_embeddings = self.negative_embedding_compressors[concept["idx"]](negative_sub_concept_embeddings)

            mixed_embeddings = (
                positive_embeddings * torch.unsqueeze(concept_probs[:, concept["idx"]], dim=-1) +
                negative_embeddings * (1 - torch.unsqueeze(concept_probs[:, concept["idx"]], dim=-1))
            )

            return mixed_embeddings

        bottleneck = []
        for top_level_concept in self.concepts:
            bottleneck.append(mix_embeddings_recursive(top_level_concept))
        bottleneck = torch.concatenate(bottleneck, dim=1)

        return bottleneck

    def forward(self, x, c_true=None, train=False):
        # Calculate concept probabilities and embeddings
        positive_and_negative_concept_embeddings, predicted_concept_probs = self.compute_concept_embeddings_and_probs(x)

        # Perform interventions
        intervention_mask = self.intervention_mask
        if train and c_true is not None and intervention_mask is None:
            intervention_mask = torch.bernoulli(torch.full((self.n_concepts,), 0.25))

        interventions = torch.zeros_like(predicted_concept_probs)

        if c_true is not None and intervention_mask is not None:
            c_true = torch.where(
                torch.logical_or(c_true == 0, c_true == 1),
                c_true,
                predicted_concept_probs
            )

            for concept_idx in range(self.n_concepts):
                if intervention_mask[concept_idx] == 1:
                    interventions[:, concept_idx] = 1

            interventions = self.calculate_implied_interventions(interventions, c_true)
        
        concept_probs_after_interventions = c_true * interventions + predicted_concept_probs * (1 - interventions)

        # Mix embeddings
        bottleneck = self.mix_embeddings(positive_and_negative_concept_embeddings, concept_probs_after_interventions)

        # Run task predictor
        y_logits = self.label_predictor(bottleneck)

        return {
            "predicted_concept_probs": predicted_concept_probs,
            "y_logits": y_logits,
            "bottleneck": bottleneck,
            "intervention_mask": intervention_mask,
            "positive_and_negative_concept_embeddings": positive_and_negative_concept_embeddings
        }
