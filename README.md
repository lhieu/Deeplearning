# Phân loại 36 loại rau củ quả với StarNet và CNN

**Nhóm 1** – Môn Deep Learning

| MSSV | Họ tên |
|---|---|
| 106220132 | Lê Văn Hiếu |
| 106220155 | Ngô Trung Tín |
| 106220161 | Phạm Ngọc Vũ |

## Dữ liệu

Fruit and Vegetable Image Recognition (36 lớp):
https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition

## Cấu trúc thư mục

```
├── CNN/          backbone.py, data_loader.py, train.py
├── StarNet/      backbone.py, data_loader.py, train.py
├── notebooks/    Bản sao của CNN_build.ipynb, Bản sao của Build_starnet.ipynb
└── results/      chart_cnn.png, chart_starnet.png
```

## Cấu hình huấn luyện

| Thông số | Giá trị |
|---|---|
| Kích thước ảnh | 128 × 128 |
| Batch size | 64 |
| Số epoch | 25 |
| Optimizer | AdamW (lr 1e-3, weight decay 1e-4) |
| Learning rate | Cosine Annealing |
| Augmentation | Flip, Rotate 15°, ColorJitter 0.2 |
| Mixed precision | AMP FP16 |
| Lưu mô hình theo | Macro-F1 trên tập validation |

## Cách chạy (Google Colab)

1. Tải dữ liệu từ Kaggle, đặt vào `data/train` và `data/val`.
2. Mở notebook trong `notebooks/`, chọn Runtime → GPU.
3. Chạy lần lượt các cell, hoặc chạy trực tiếp:
```
cd CNN        # hoặc cd StarNet
python train.py
```

> Repo không kèm file trọng số (.pth). Chạy `train.py` để huấn luyện lại; mô hình tốt nhất sẽ được lưu tự động.

## Kết quả

| Mô hình | Số tham số | Best Val Accuracy | Best Val Macro-F1 | GPU | Thời gian (25 epoch) |
|---|---|---|---|---|---|
| CustomFruitNet (CNN) | 1,182,468 | 83.19% | 83.00% | T4 | 51.5 phút |
| StarNetLite | 305,604 | **96.30%** | **96.30%** | L4 | 31.8 phút |

### CustomFruitNet (CNN)
![CNN](results/chart_cnn.png)

### StarNetLite
![StarNet](results/chart_starnet.png)

## Nhận xét

- StarNetLite đạt kết quả cao hơn CNN khoảng 13% dù chỉ dùng khoảng 1/4 số tham số.
- Train accuracy của CNN ở epoch cuối là 62.67% và vẫn đang tăng, nên nếu tăng số epoch có thể còn cải thiện.

## Tham khảo

Ma, X. et al. *Rewrite the Stars*, CVPR 2024.
