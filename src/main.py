from models.resnet34 import resnet34
from data.augment_data import setup_data_loaders
from training.trainer import train_model, evaluate_model
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = resnet34(pretrained=True, num_classes=15)
    model = model.to(device)

    base_path = "/home/artur_176/CNN/CNN/datasets/processed"
    train_loader, val_loader, test_loader = setup_data_loaders(base_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 25
    best_model_path = "best_model.pth"
    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs,
        device,
        best_model_path,
    )

    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
