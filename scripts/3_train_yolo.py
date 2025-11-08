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

# KONFIGURASI RINGAN (untuk CPU) DENGAN AUGMENTASI
EPOCH = 150             # Sedikit lebih banyak epoch untuk augmentasi
BATCH_SIZE = 16         # Sesuaikan jika VRAM tidak cukup
UKURAN_GAMBAR = 416     # Ukuran lebih kecil, lebih cepat
PATIENCE = 20           # Berhenti jika tidak ada peningkatan setelah 20 epoch

# ============================================
# SETUP
# ============================================

def hapus_folder_jika_ada(folder_path):
    """Menghapus folder jika sudah ada untuk memastikan kebersihan."""
    if os.path.exists(folder_path):
        import shutil
        shutil.rmtree(folder_path)
        print(f"🗑️  Folder lama dihapus: {folder_path}")

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

# Hapus folder hasil training lama jika ada
hapus_folder_jika_ada(os.path.join(FOLDER_RESULTS, 'kersen_v2'))

# ============================================
# TRAINING
# ============================================

print("🚀 Memulai training...")
print(f"Data: {PATH_DATA_YAML}")
print(f"Epoch: {EPOCH}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Ukuran gambar: {UKURAN_GAMBAR}")
print(f"Device: {device}")
print(f"⏱️  Estimasi waktu: 15-30 menit di CPU\n")

# Load model
model = YOLO('yolov8s.pt') # Menggunakan model nano yang paling ringan

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
    optimizer='Adam', # Menggunakan Adam optimizer
    lr0=0.001, # Learning rate yang umum untuk Adam
    plots=True,
    # Augmentasi yang lebih kuat
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=20.0,      # Rotasi gambar hingga 20 derajat
    translate=0.2,     # Geser gambar hingga 20%
    scale=0.7,         # Skala gambar antara 0.3 dan 1.7
    flipud=0.5,        # Balik vertikal 50%
    fliplr=0.5,        # Balik horizontal 50%
    mosaic=1.0,        # Gabungkan 4 gambar menjadi 1
    mixup=0.1,         # Campur 2 gambar (sangat efektif untuk generalisasi)
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