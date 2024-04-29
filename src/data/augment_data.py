from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os


def get_train_transforms():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_test_val_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def setup_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_data(data_path, transforms):
    dataset = ImageFolder(root=data_path, transform=transforms)
    print(f"Loaded {len(dataset)} images under {data_path}")
    return DataLoader(dataset, batch_size=1, shuffle=True)


def setup_data_loaders(base_path):
    train_loader = load_data(os.path.join(base_path, "train"), get_train_transforms())
    val_loader = load_data(os.path.join(base_path, "val"), get_test_val_transforms())
    test_loader = load_data(os.path.join(base_path, "test"), get_test_val_transforms())
    return train_loader, val_loader, test_loader
