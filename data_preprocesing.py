from torchvision import transforms, datasets
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
from sklearn.datasets import fetch_lfw_people
from PIL import Image
import os


class DatasetLoader(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label


def prepare_nn_data_lfw(top_classes=40):
    return _prepare_nn_data(data_dir='dataset/lfw_funneled', top_classes=top_classes)


def prepare_nn_data_cwf(top_classes=40):
    return _prepare_nn_data(data_dir='dataset/casia-webface', top_classes=top_classes)


def prepare_ml_data_lfw(top_classes=40):
    return _prepare_ml_data(data_dir='dataset/lfw_funneled', top_classes=top_classes)


def prepare_ml_data_cwf(top_classes=40):
    return _prepare_ml_data(data_dir='dataset/casia-webface', top_classes=top_classes)


def _prepare_nn_data(data_dir, top_classes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    loaded_dataset = DatasetLoader(root_dir=data_dir, transform=transform)
    _filter_dataset(loaded_dataset, top_classes)
    return loaded_dataset, loaded_dataset.classes


def _prepare_ml_data(data_dir, top_classes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    loaded_dataset = DatasetLoader(root_dir=data_dir, transform=transform)
    _filter_dataset(loaded_dataset, top_classes)

    data = []
    targets = []
    for img, label in loaded_dataset:
        data.append(img.numpy().flatten())
        targets.append(label)

    data = np.array(data)
    targets = np.array(targets)
    _, h, w = loaded_dataset[0][0].shape

    return data, targets, h, w


# If needed for stratification
def _filter_dataset(dataset, top_classes):
    class_counter = Counter([label for _, label in dataset])
    top_classes_counts = class_counter.most_common(top_classes)
    top_classes_indices = [label for label, _ in top_classes_counts]
    filtered_samples = [(img, label) for img, label in dataset if label in top_classes_indices]
    dataset.dataset.samples = filtered_samples
    dataset.dataset.targets = [label for _, label in filtered_samples]
    dataset.classes = [dataset.classes[label] for label in top_classes_indices]


if __name__ == '__main__':
    dataset, classes = prepare_nn_data_cwf()
    print(f'Found {len(dataset)} images in {len(classes)} classes.')
    X, y, _, _ = prepare_ml_data()
    print(f'Prepared data for mlL {X.shape}, {y.shape}')
