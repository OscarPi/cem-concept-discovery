import numpy as np
import sklearn.model_selection
import torch
import torchvision

DATASET_DIR = "datasets"

x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []
def load_mnist(root_dir=DATASET_DIR):
    global x_train, y_train, x_val, y_val, x_test, y_test
    if len(x_train) > 0:
        return

    train_val_ds = torchvision.datasets.MNIST(root_dir, train=True, download=True)
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

    test_ds = torchvision.datasets.MNIST(root_dir, train=False, download=True)
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
    def __init__(self, n_digits, selected_digits=(0, 1), root_dir=DATASET_DIR):
        self.n_digits = n_digits
        self.selected_digits = selected_digits

        if len(x_train) == 0:
            load_mnist(root_dir)

        np.random.seed(42)
        self.train_samples, self.train_labels = create_addition_set(x_train, y_train, n_digits, selected_digits, 10000)
        self.val_samples, self.val_labels = create_addition_set(x_val, y_val, n_digits, selected_digits, int(10000 * 0.2))
        self.test_samples, self.test_labels = create_addition_set(x_test, y_test, n_digits, selected_digits, 10000)

    def get_concepts(self, labels, concepts):
        c = []
        for d in range(self.n_digits):
            if len(self.selected_digits) <= 2 and (concepts is None or concepts[d]):
                c.append(labels[:, d] == self.selected_digits[1])
            elif len(self.selected_digits) > 2:
                for i in range(len(self.selected_digits)):
                    if concepts is None or concepts[d][i]:
                        c.append(labels[:, d] == self.selected_digits[i])
        return torch.FloatTensor(np.stack(c, axis=-1))

    def get_labels(self, labels, threshold=None):
        if threshold is None:
            return torch.LongTensor(np.sum(labels, axis=1))

        return torch.FloatTensor(np.sum(labels, axis=1) >= threshold)

    def train_dl(self, concepts=None, threshold=None, concept_generator=None, additional_concepts=None):
        if concept_generator is None:
            c = self.get_concepts(self.train_labels, concepts)
        else:
            c = concept_generator(self.train_labels)
        c = torch.FloatTensor(c)
        if additional_concepts is not None:
            c = torch.cat((c, torch.FloatTensor(additional_concepts)), dim=1)
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.train_samples),
            self.get_labels(self.train_labels, threshold),
            c
        )
        return torch.utils.data.DataLoader(dataset, batch_size=2048, num_workers=7)

    def val_dl(self, concepts=None, threshold=None, concept_generator=None, additional_concepts=None):
        if concept_generator is None:
            c = self.get_concepts(self.val_labels, concepts)
        else:
            c = concept_generator(self.val_labels)
        c = torch.FloatTensor(c)
        if additional_concepts is not None:
            c = torch.cat((c, torch.FloatTensor(additional_concepts)), dim=1)
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.val_samples),
            self.get_labels(self.val_labels, threshold),
            c
        )
        return torch.utils.data.DataLoader(dataset, batch_size=2048, num_workers=7)

    def test_dl(self, concepts=None, threshold=None, concept_generator=None, additional_concepts=None):
        if concept_generator is None:
            c = self.get_concepts(self.test_labels, concepts)
        else:
            c = concept_generator(self.test_labels)
        c = torch.FloatTensor(c)
        if additional_concepts is not None:
            c = torch.cat((c, torch.FloatTensor(additional_concepts)), dim=1)
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.test_samples),
            self.get_labels(self.test_labels, threshold),
            c
        )
        return torch.utils.data.DataLoader(dataset, batch_size=2048, num_workers=7)
