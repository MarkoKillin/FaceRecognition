from torch import save
from sklearn.datasets import fetch_lfw_people
from torchvision import transforms
from torch.utils.data import DataLoader, DataSet
import numpy as np


class LFWDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def prepare_data():
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    X = lfw_people.images[:, :, :, np.newaxis] / 255.0
    y = lfw_people.target

    transform = transforms.Compose([
        transforms.toPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = LFWDataset(X, y, transform=transform)
    return dataset, lfw_people.target_names


if __name__ == '__main__':
    dataset, target_names = prepare_date()
    print('----data loaded----')
