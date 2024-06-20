from torchvision import transforms, datasets
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
from sklearn.datasets import fetch_lfw_people
from PIL import Image


class LFWDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = fetch_lfw_people(data_home=root_dir, min_faces_per_person=40, download_if_missing=False)
        self.classes = self.dataset.target_names

    def __len__(self):
        return len(self.dataset.images)

    def __getitem__(self, idx):
        img = self.dataset.images[idx]
        label = self.dataset.target[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label


class ORLFacesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir


def prepare_nn_data(data_dir='dataset/lfw_funneled'):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    lfw_dataset = LFWDataset(root_dir=data_dir, transform=transform)

    return lfw_dataset, lfw_dataset.classes


def prepare_ml_data(data_dir='dataset/lfw_funneled'):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    lfw_dataset = fetch_lfw_people(data_home=data_dir, min_faces_per_person=100, download_if_missing=True)
    _, h, w = lfw_dataset.images.shape
    return lfw_dataset.data, lfw_dataset.target, h, w


# If needed for stratification
def filter_dataset(dataset, min_instances=2):
    classes = dataset.classes
    num_classes = Counter(classes)
    filtered_indicies = [i for i, label in enumerate(classes) if num_classes[label] >= min_instances]
    filtered_samples = [dataset.samples[i] for i in filtered_indicies]
    dataset.samples = filtered_samples
    dataset.targets = [sample[1] for sample in filtered_samples]
    return dataset


if __name__ == '__main__':
    dataset, classes = prepare_nn_data()
    print(f'Found {len(dataset)} images in {len(classes)} classes.')
    X, y, _, _ = prepare_ml_data()
    print(f'Prepared data for mlL {X.shape}, {y.shape}')
