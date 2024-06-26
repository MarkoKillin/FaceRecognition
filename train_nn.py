import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from data_preprocesing import prepare_nn_data_lfw, prepare_nn_data_cwf
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split


def train_model(model, train_loader, val_loader, device, epochs=10, save_path='models/model.pth', log_path='runs/'):
    print('----Starting training----')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    writer = SummaryWriter(log_dir=log_path)

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{epochs}: Loss: {epoch_loss:.4f}')

        writer.add_scalar('Loss/train', epoch_loss, epoch)

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')
    writer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train neural network model')
    parser.add_argument('--model', type=str, default='resnet50', choices=['cnn', 'resnet18', 'resnet50'])
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    args = parser.parse_args()

    dataset, classes = prepare_nn_data_cwf()

    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader_1 = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader_1 = DataLoader(val_dataset, batch_size=32, shuffle=True)
    print('----Data Loaded----')

    train_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training on: ' + 'cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = len(classes)
    print(f'Number of classes: {num_classes}')

    if args.model == 'resnet18':
        from model_resnet18 import get_resnet_model

        sel_model = get_resnet_model(num_classes)
    elif args.model == 'resnet50':
        from model_resnet50 import get_resnet_model

        sel_model = get_resnet_model(num_classes)
    else:
        from model_cnn import get_cnn_model

        sel_model = get_cnn_model(num_classes)

    sel_model.to(train_device)
    train_model(sel_model, train_loader_1, val_loader_1, train_device, epochs=args.epochs,
                save_path=f'models/model_{args.model}.pth', log_path=f'runs/model_{args.model}')
