from torchvision import transforms, datasets
from torch.utils.data import Dataset


class LFWDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(root=root_dir, transform=self.transform)
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label


def prepare_data(data_dir='dataset/lfw_funneled'):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    lfw_dataset = LFWDataset(root_dir=data_dir, transform=transform)
    return lfw_dataset, lfw_dataset.classes


if __name__ == '__main__':
    dataset, classes = prepare_data()
    print(f"Found {len(dataset)} images in {len(classes)} classes.")
