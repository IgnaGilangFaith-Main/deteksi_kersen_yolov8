"""
Script untuk test model dengan foto dari folder test
Menampilkan metrik evaluasi dan grafik
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score,
    accuracy_score,
    classification_report
)
import seaborn as sns

# ============================================
# KONFIGURASI
# ============================================

PATH_MODEL = r"models/yolov8n_kersen_best.pt"
PATH_TEST = r"train_split/images/test"
PATH_LABELS_TEST = r"train_split/labels/test"
FOLDER_OUTPUT = r"results/inference_output"
CONFIDENCE_THRESHOLD = 0.5

# ============================================
# SETUP
# ============================================

Path(FOLDER_OUTPUT).mkdir(parents=True, exist_ok=True)

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

# Cek model
if not os.path.exists(PATH_MODEL):
    print(f"❌ Model tidak ditemukan: {PATH_MODEL}")
    print("⚠️  Jalankan script training terlebih dahulu!")
    exit(1)

# Load model
print("📂 Memuat model...")
model = YOLO(PATH_MODEL)
print("✅ Model berhasil dimuat\n")

# Nama kelas
NAMA_KELAS = ['mentah', 'setengah_matang', 'matang']
NAMA_KELAS_DISPLAY = ['Mentah', 'Setengah Matang', 'Matang']

# ============================================
# FUNGSI HELPER
# ============================================

def baca_label_yolo(path_label):
    """Membaca label YOLO format"""
    labels = []
    if os.path.exists(path_label):
        with open(path_label, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = int(parts[0])
                    labels.append(class_id)
    return labels

def ambil_ground_truth(path_labels):
    """Ambil ground truth labels dari folder"""
    y_true = []
    
    file_labels = sorted([f for f in os.listdir(path_labels) if f.endswith('.txt')])
    
    for label_file in file_labels:
        path_label = os.path.join(path_labels, label_file)
        labels = baca_label_yolo(path_label)
        
        if len(labels) > 0:
            y_true.append(labels[0])
    
    return y_true

# ============================================
# INFERENCE
# ============================================

print("=" * 50)
print("INFERENCE & EVALUASI - DETEKSI KEMATANGAN KERSEN")
print("=" * 50)

# Ambil ground truth
print("\n📊 Mengambil ground truth labels...")
y_true = ambil_ground_truth(PATH_LABELS_TEST)
print(f"✅ Ground truth: {len(y_true)} gambar")

# Deteksi semua gambar
print("\n🎯 Menjalankan inference...")
hasil = model.predict(
    source=PATH_TEST,
    conf=CONFIDENCE_THRESHOLD,
    device=device,
    verbose=False
)

print(f"✅ Ditemukan {len(hasil)} gambar\n")

# ============================================
# PROSES HASIL DETEKSI
# ============================================

y_pred = []
deteksi_info = []

for idx, result in enumerate(hasil):
    nama_file = os.path.basename(result.path)
    
    print(f"📷 Gambar {idx+1}: {nama_file}")
    print(f"   Total deteksi: {len(result.boxes)}")
    
    # Plot hasil
    im_array = result.plot()
    
    # Simpan hasil
    path_output = os.path.join(FOLDER_OUTPUT, f'hasil_{idx:04d}.jpg')
    cv2.imwrite(path_output, im_array)
    print(f"   Disimpan: {path_output}")
    
    # Ambil prediksi
    if len(result.boxes) > 0:
        best_box = result.boxes[0]
        id_kelas = int(best_box.cls[0])
        conf = float(best_box.conf[0])
        nama_kelas = NAMA_KELAS_DISPLAY[id_kelas]
        
        y_pred.append(id_kelas)
        
        print(f"   Prediksi: {nama_kelas} ({conf:.2f})")
        
        deteksi_info.append({
            'file': nama_file,
            'kelas': nama_kelas,
            'confidence': conf
        })
    else:
        y_pred.append(-1)
        print(f"   Prediksi: Tidak ada deteksi")
    
    print()

# ============================================
# HITUNG METRIK EVALUASI
# ============================================

print("=" * 50)
print("EVALUASI METRIK")
print("=" * 50)

# Filter out -1 (tidak terdeteksi)
mask = np.array(y_pred) != -1
y_true_filtered = np.array(y_true)[mask]
y_pred_filtered = np.array(y_pred)[mask]

if len(y_true_filtered) == 0:
    print("\n⚠️  Tidak ada deteksi yang valid!")
    exit(1)

# Hitung metrik
accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
precision = precision_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
recall = recall_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
f1 = f1_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)

# Confusion matrix
cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=[0, 1, 2])

# Classification report
class_report = classification_report(y_true_filtered, y_pred_filtered, 
                                    target_names=NAMA_KELAS_DISPLAY, 
                                    zero_division=0)

# Tampilkan metrik
print(f"\n✅ Akurasi:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"✅ Presisi:  {precision:.4f}")
print(f"✅ Recall:   {recall:.4f}")
print(f"✅ F-1 Score: {f1:.4f}")

print(f"\n📊 Classification Report:")
print(class_report)

print(f"\n📊 Confusion Matrix:")
print(cm)

# ============================================
# BUAT GRAFIK
# ============================================

print("\n🎨 Membuat grafik...")

# Figure 1: Metrik Bar Chart
fig1, ax1 = plt.subplots(figsize=(10, 6))

metrik_names = ['Akurasi', 'Presisi', 'Recall', 'F-1 Score']
metrik_values = [accuracy, precision, recall, f1]
colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']

bars = ax1.bar(metrik_names, metrik_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Tambah value di atas bar
for bar, value in zip(bars, metrik_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Metrik Evaluasi Model Deteksi Kersen', fontsize=14, fontweight='bold', pad=20)
ax1.set_ylim([0, 1.1])
ax1.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
path_metrik = os.path.join(FOLDER_OUTPUT, 'metrik_evaluasi.png')
plt.savefig(path_metrik, dpi=300, bbox_inches='tight')
print(f"✅ Grafik metrik disimpan: {path_metrik}")
plt.close()

# Figure 2: Confusion Matrix Heatmap
fig2, ax2 = plt.subplots(figsize=(8, 7))

sns.heatmap(cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=NAMA_KELAS_DISPLAY,
            yticklabels=NAMA_KELAS_DISPLAY,
            cbar_kws={'label': 'Count'},
            ax=ax2,
            linewidths=1,
            linecolor='black')

ax2.set_xlabel('Prediksi', fontsize=12, fontweight='bold')
ax2.set_ylabel('Ground Truth', fontsize=12, fontweight='bold')
ax2.set_title('Confusion Matrix - Deteksi Kematangan Kersen', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
path_cm = os.path.join(FOLDER_OUTPUT, 'confusion_matrix.png')
plt.savefig(path_cm, dpi=300, bbox_inches='tight')
print(f"✅ Confusion matrix disimpan: {path_cm}")
plt.close()

# Figure 3: Per-Class Metrics
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Hitung per-class metrics
precision_per_class = precision_score(y_true_filtered, y_pred_filtered, 
                                      labels=[0, 1, 2], average=None, zero_division=0)
recall_per_class = recall_score(y_true_filtered, y_pred_filtered, 
                                labels=[0, 1, 2], average=None, zero_division=0)
f1_per_class = f1_score(y_true_filtered, y_pred_filtered, 
                        labels=[0, 1, 2], average=None, zero_division=0)

x = np.arange(len(NAMA_KELAS_DISPLAY))
width = 0.25

bars1 = ax3.bar(x - width, precision_per_class, width, label='Presisi', alpha=0.8, color='#667eea')
bars2 = ax3.bar(x, recall_per_class, width, label='Recall', alpha=0.8, color='#764ba2')
bars3 = ax3.bar(x + width, f1_per_class, width, label='F-1 Score', alpha=0.8, color='#f093fb')

ax3.set_xlabel('Kelas', fontsize=12, fontweight='bold')
ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
ax3.set_title('Metrik Per-Class', fontsize=14, fontweight='bold', pad=20)
ax3.set_xticks(x)
ax3.set_xticklabels(NAMA_KELAS_DISPLAY)
ax3.legend(fontsize=11)
ax3.set_ylim([0, 1.1])
ax3.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
path_per_class = os.path.join(FOLDER_OUTPUT, 'metrik_per_class.png')
plt.savefig(path_per_class, dpi=300, bbox_inches='tight')
print(f"✅ Per-class metrics disimpan: {path_per_class}")
plt.close()

# ============================================
# SIMPAN LAPORAN
# ============================================

print("\n💾 Menyimpan laporan...")

path_laporan = os.path.join(FOLDER_OUTPUT, 'laporan_evaluasi.txt')
with open(path_laporan, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("LAPORAN INFERENCE & EVALUASI - DETEKSI KEMATANGAN KERSEN\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("METRIK EVALUASI:\n")
    f.write("-" * 60 + "\n")
    f.write(f"Akurasi:     {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Presisi:     {precision:.4f}\n")
    f.write(f"Recall:      {recall:.4f}\n")
    f.write(f"F-1 Score:   {f1:.4f}\n\n")
    
    f.write("CLASSIFICATION REPORT:\n")
    f.write("-" * 60 + "\n")
    f.write(class_report + "\n\n")
    
    f.write("CONFUSION MATRIX:\n")
    f.write("-" * 60 + "\n")
    f.write("                Mentah  Setengah  Matang\n")
    for i, class_name in enumerate(NAMA_KELAS_DISPLAY):
        f.write(f"{class_name:15} {cm[i][0]:4d}    {cm[i][1]:4d}     {cm[i][2]:4d}\n")

print(f"✅ Laporan disimpan: {path_laporan}")

# ============================================
# RINGKASAN
# ============================================

print("\n" + "=" * 50)
print("RINGKASAN HASIL")
print("=" * 50)
print(f"\n📊 Total gambar test: {len(y_true)}")
print(f"✅ Akurasi: {accuracy*100:.2f}%")
print(f"✅ Presisi: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F-1 Score: {f1:.4f}")

print(f"\n📁 Hasil disimpan di: {FOLDER_OUTPUT}")
print(f"   - hasil_0001.jpg, hasil_0002.jpg, ... (inference results)")
print(f"   - metrik_evaluasi.png (grafik metrik)")
print(f"   - confusion_matrix.png (confusion matrix)")
print(f"   - metrik_per_class.png (metrik per kelas)")
print(f"   - laporan_evaluasi.txt (laporan lengkap)")

print("\n✨ Inference & evaluasi selesai!")