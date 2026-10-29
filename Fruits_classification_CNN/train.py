import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support

from backbone import CustomFruitNet
from data_loader import get_data_loaders


def count_parameters(model):
    """Tính tổng số tham số có thể huấn luyện (Trainable Parameters)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass với Mixed Precision (FP16)
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass với GradScaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    val_loss = running_loss / total

    correct = sum([p == l for p, l in zip(all_preds, all_labels)])
    val_acc = correct / total

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    return val_loss, val_acc, precision, recall, f1


def main():
    # 1. Cấu hình siêu tham số
    data_dir = "data"
    batch_size = 64
    num_classes = 36
    num_epochs = 25
    learning_rate = 1e-3
    num_workers = 2  # Phù hợp cấu hình 2 vCPU của Colab, tránh nghẽn thread

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Thiết bị sử dụng: {device}")
    
    if device.type == "cuda":
        print(f"--> Tên GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # 2. Khởi tạo DataLoader
    train_loader, val_loader, class_names = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=(128, 128)
    )
    print(f"--> Nhận diện được {len(class_names)} lớp trái cây.")

    # 3. Khởi tạo mô hình
    model = CustomFruitNet(num_classes=num_classes).to(device)
    total_params = count_parameters(model)
    print(f"--> Tổng tham số CustomFruitNet: {total_params:,} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Scaler cho Mixed Precision
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_f1 = 0.0
    start_time = time.time()

    print("\n" + "=" * 70)
    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()

        epoch_duration = time.time() - epoch_start

        print(
            f"Epoch [{epoch + 1:02d}/{num_epochs:02d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc * 100:.2f}% | "
            f"Macro-F1: {val_f1 * 100:.2f}%"
        )

        # Lưu checkpoint theo Macro F1 cao nhất
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_custom_fruitnet.pth")
            print(f"[*] Checkpoint saved: New best Macro-F1 = {best_f1 * 100:.2f}%")

    total_time = (time.time() - start_time) / 60
    print("=" * 70)
    print(f"--> Huấn luyện hoàn tất trong {total_time:.2f} phút. Best F1: {best_f1 * 100:.2f}%")


if __name__ == "__main__":
    main()