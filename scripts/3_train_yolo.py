"""
Script untuk training YOLOv8 dengan dataset kersen
Gunakan GPU jika tersedia untuk hasil lebih cepat
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch

# ============================================
# KONFIGURASI
# ============================================

PATH_DATA_YAML = r"train_split/data.yaml"
FOLDER_MODELS = r"models"
FOLDER_RESULTS = r"results"

EPOCH = 100
BATCH_SIZE = 16  # Kurangi jika out of memory
UKURAN_GAMBAR = 640
PATIENCE = 20  # Early stopping

# ============================================
# SETUP
# ============================================

Path(FOLDER_MODELS).mkdir(exist_ok=True)
Path(FOLDER_RESULTS).mkdir(exist_ok=True)

# Cek device yang tersedia
print("\n" + "=" * 50)
print("INFO PERANGKAT")
print("=" * 50)
print(f"GPU Tersedia: {torch.cuda.is_available()}")
print(f"Jumlah GPU: {torch.cuda.device_count()}")

# Gunakan CPU jika GPU tidak tersedia
if torch.cuda.is_available():
    device = 0
    device_name = "GPU"
    print(f"GPU yang digunakan: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    device_name = "CPU"
    print(f"Menggunakan: {device_name}")

print("=" * 50 + "\n")

# ============================================
# TRAINING
# ============================================

print(f"\n🚀 Memulai training...")
print(f"Data: {PATH_DATA_YAML}")
print(f"Epoch: {EPOCH}")
print(f"Batch size: {BATCH_SIZE}")

# Load model
model = YOLO('yolov8n.pt')  # Model Nano (cepat & efisien)

# Training
hasil = model.train(
    data=PATH_DATA_YAML,
    epochs=EPOCH,
    imgsz=UKURAN_GAMBAR,
    batch=BATCH_SIZE,
    patience=PATIENCE,
    device=device,  # Gunakan device yang sudah ditentukan
    project=FOLDER_RESULTS,
    name='kersen_v1',
    save=True,
    verbose=True,
    pretrained=True,
    optimizer='SGD',
    momentum=0.937,
    weight_decay=0.0005,
    lr0=0.01,
    plots=True,
)

# ============================================
# SIMPAN MODEL
# ============================================

print("\n💾 Menyimpan model terbaik...")
path_best_model = os.path.join(FOLDER_RESULTS, 'kersen_v1', 'weights', 'best.pt')
if os.path.exists(path_best_model):
    path_simpan = os.path.join(FOLDER_MODELS, 'yolov8n_kersen_best.pt')
    import shutil
    shutil.copy2(path_best_model, path_simpan)
    print(f"✅ Model disimpan: {path_simpan}")

# ============================================
# EVALUASI
# ============================================

print("\n📊 Evaluasi pada test set...")
model = YOLO(path_best_model)

# Validasi
metrik = model.val(
    data=PATH_DATA_YAML,
    device=device,  # Gunakan device yang sudah ditentukan
)

print(f"\n✅ Training selesai!")
print(f"Hasil disimpan di: {FOLDER_RESULTS}")