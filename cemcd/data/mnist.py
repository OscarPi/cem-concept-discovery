import numpy as np
import sklearn.model_selection
import torch
import torchvision

x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []
def load_mnist(dataset_dir):
    global x_train, y_train, x_val, y_val, x_test, y_test
    if len(x_train) > 0:
        return

    train_val_ds = torchvision.datasets.MNIST(dataset_dir, train=True, download=True)
    x_train_val = []
    y_train_val = []
    for x, y in train_val_ds:
        x_train_val.append(np.expand_dims(x, axis=(0, 1)))
        y_train_val.append(np.expand_dims(y, axis=0))
    x_train_val = np.concatenate(x_train_val, axis=0)
    y_train_val = np.concatenate(y_train_val, axis=0)

    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
        x_train_val, y_train_val, test_size=0.2, random_state=42
    )

    test_ds = torchvision.datasets.MNIST(dataset_dir, train=False, download=True)
    for x, y in test_ds:
        x_test.append(np.expand_dims(x, axis=(0, 1)))
        y_test.append(np.expand_dims(y, axis=0))
    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

def create_addition_set(x, y, n_digits, selected_digits, dataset_size):
    x = x[np.isin(y, selected_digits)]
    y = y[np.isin(y, selected_digits)]
    samples = []
    labels = []
    for _ in range(dataset_size):
        digits = []
        digit_labels = []
        for _ in range(n_digits):
            idx = np.random.choice(x.shape[0])
            digits.append(x[idx:idx+1].copy())
            digit_labels.append(y[idx:idx+1, None])
        samples.append(np.concatenate(digits, axis=1))
        labels.append(np.concatenate(digit_labels, axis=1))
    samples = np.concatenate(samples, axis=0)
    labels = np.concatenate(labels, axis=0)
    samples = samples.astype(np.float64) / 255.0
    return samples, labels

class MNISTDatasets:
    def __init__(self, n_digits, max_digit, dataset_dir):
        selected_digits = tuple(range(max_digit + 1))

        if len(x_train) == 0:
            load_mnist(dataset_dir)

        np.random.seed(42)
        self.train_samples, self.train_labels = create_addition_set(x_train, y_train, n_digits, selected_digits, 10000)
        self.val_samples, self.val_labels = create_addition_set(x_val, y_val, n_digits, selected_digits, int(10000 * 0.2))
        self.test_samples, self.test_labels = create_addition_set(x_test, y_test, n_digits, selected_digits, 10000)

        concept_bank = []
        concept_test_ground_truth = []
        self.concept_names = []

        for i in range(n_digits):
            for j in selected_digits:
                concept_bank.append(self.train_labels[:, i] == j)
                concept_test_ground_truth.append(self.test_labels[:, i] == j)
                self.concept_names.append(f"Digit {i} is {j}")

        self.concept_bank = np.stack(concept_bank, axis=1)
        self.concept_test_ground_truth = np.stack(concept_test_ground_truth, axis=1)

        self.concept_generator = lambda labels: labels > (max_digit / 2)
        self.n_concepts = n_digits
        self.n_tasks = max_digit * n_digits + 1

    def get_labels(self, labels):
        return torch.LongTensor(np.sum(labels, axis=1))

    def train_dl(self, additional_concepts=None):
        c = self.concept_generator(self.train_labels)
        c = torch.FloatTensor(c)
        if additional_concepts is not None:
            c = torch.cat((c, torch.FloatTensor(additional_concepts)), dim=1)
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.train_samples),
            self.get_labels(self.train_labels),
            c
        )
        return torch.utils.data.DataLoader(dataset, batch_size=2048, num_workers=7)

    def val_dl(self, additional_concepts=None):
        c = self.concept_generator(self.val_labels)
        c = torch.FloatTensor(c)
        if additional_concepts is not None:
            c = torch.cat((c, torch.FloatTensor(additional_concepts)), dim=1)
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.val_samples),
            self.get_labels(self.val_labels),
            c
        )
        return torch.utils.data.DataLoader(dataset, batch_size=2048, num_workers=7)

    def test_dl(self, additional_concepts=None):
        c = self.concept_generator(self.test_labels)
        c = torch.FloatTensor(c)
        if additional_concepts is not None:
            c = torch.cat((c, torch.FloatTensor(additional_concepts)), dim=1)
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.test_samples),
            self.get_labels(self.test_labels),
            c
        )
        return torch.utils.data.DataLoader(dataset, batch_size=2048, num_workers=7)
