import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def convert_to_rgb(img):
    """Ép ảnh về chuẩn RGB 3 kênh, tránh lỗi palette và độ trong suốt"""
    return img.convert('RGB')

def get_transforms(img_size=(128, 128)):
    train_transform = transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(convert_to_rgb),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def get_data_loaders(data_dir="/content/data", batch_size=64, num_workers=2, img_size=(128, 128)):
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")

    train_transform, val_transform = get_transforms(img_size=img_size)

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

    # num_workers=2 là tối ưu cho Colab, persistent_workers giữ worker sống qua các epoch
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, train_dataset.classes