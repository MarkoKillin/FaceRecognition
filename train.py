from torch.optim import optim
from torch.nn import nn
from torch.utils.data import DataLoader
from data_preprocessing import prepare_data, LFWDataset
import os


def train_model(model, train_loader, epochs, save_path='models/model.pth'):
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

    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'resnet', 'efficientnet'])
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    args = parser.parse_args()

    dataset, target_names = prepare_data()
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    num_classes = len(target_names)

    if args.model == 'resnet':
        from model_resnet import get_resnet_model
        model = get_resnet_model(num_classes)
    elif args.model == 'efficientnet':
        from model_efficientnet import get_efficientnet_model
        model = get_efficientnet_model(num_classes)
    else:
        from model_cnn import get_cnn_model
        model = get_cnn_model(num_classes)

    os.makedirs('models', exist_ok=True)
    train_model(model, train_loader, epochs=args.epochs, save_path=f'models/model_{args.model}.pth')
