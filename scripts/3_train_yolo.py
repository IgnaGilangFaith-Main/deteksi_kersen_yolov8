"""
Script untuk training YOLOv8 dengan dataset kersen
Versi cepat untuk CPU
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

# KONFIGURASI CEPAT (untuk CPU)
EPOCH = 100              
BATCH_SIZE = 8          
UKURAN_GAMBAR = 416     
PATIENCE = 0            

# ============================================
# SETUP
# ============================================

Path(FOLDER_MODELS).mkdir(exist_ok=True)
Path(FOLDER_RESULTS).mkdir(exist_ok=True)

# Cek device
print("\n" + "=" * 50)
print("INFO PERANGKAT")
print("=" * 50)

gpu_tersedia = torch.cuda.is_available()
jumlah_gpu = torch.cuda.device_count()

print(f"GPU Tersedia: {gpu_tersedia}")
print(f"Jumlah GPU: {jumlah_gpu}")

if gpu_tersedia:
    device = 0
    print(f"GPU yang digunakan: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print(f"Menggunakan: CPU")

print("=" * 50 + "\n")

# ============================================
# TRAINING
# ============================================

print("🚀 Memulai training...")
print(f"Data: {PATH_DATA_YAML}")
print(f"Epoch: {EPOCH}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Ukuran gambar: {UKURAN_GAMBAR}")
print(f"Device: {device}")
print(f"⏱️  Estimasi waktu: 10-20 menit di CPU\n")

# Load model
model = YOLO('yolov8s.pt')

# Training
hasil = model.train(
    data=PATH_DATA_YAML,
    epochs=EPOCH,
    imgsz=UKURAN_GAMBAR,
    batch=BATCH_SIZE,
    patience=PATIENCE,
    device=device,
    project=FOLDER_RESULTS,
    name='kersen_v2',
    save=True,
    verbose=True,
    pretrained=True,
    optimizer='SGD',
    momentum=0.937,
    weight_decay=0.0005,
    lr0=0.01,
    plots=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
)

# ============================================
# SIMPAN MODEL
# ============================================

print("\n💾 Menyimpan model terbaik...")
path_best_model = os.path.join(FOLDER_RESULTS, 'kersen_v2', 'weights', 'best.pt')

if os.path.exists(path_best_model):
    path_simpan = os.path.join(FOLDER_MODELS, 'yolov8n_kersen_best.pt')
    import shutil
    shutil.copy2(path_best_model, path_simpan)
    print(f"✅ Model disimpan: {path_simpan}")
else:
    print(f"⚠️  File model tidak ditemukan: {path_best_model}")

# ============================================
# EVALUASI
# ============================================

print("\n📊 Evaluasi pada test set...")
model = YOLO(path_best_model)

metrik = model.val(
    data=PATH_DATA_YAML,
    device=device,
)

print(f"\n✅ Training selesai!")
print(f"Hasil disimpan di: {FOLDER_RESULTS}")