from torch.optim import optim
from torch.nn import nn
from torch.utils.data import DataLoader
from model import get_model
from data_preprocessing import prepare_data, LFWDataset


def train_model(model, train_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}: Loss: {epoch_loss:.4f}')


if __name__ == '__main__':
    dataset, target_names = prepare_data()
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    num_classes = len(target_names)
    model = get_model()

    train_model(model, train_loader, epochs=10)