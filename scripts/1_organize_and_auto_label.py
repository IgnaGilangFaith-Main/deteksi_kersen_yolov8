"""
Script untuk mengorganisir foto dan auto-label
Dari: datasets/ 
Ke: dataset_organized/ dengan format YOLO
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

# ============================================
# KONFIGURASI
# ============================================

# Path ke folder datasets
DATASETS_PATH = r"datasets"

# Path output (akan dibuat otomatis)
OUTPUT_PATH = r"dataset_organized"

# Mapping folder ke class ID
KELAS_MAPPING = {
    "mentah": 0,           # Belum matang
    "setengah_matang": 1,  # Setengah matang
    "matang": 2            # Matang
}

# ============================================
# FUNGSI HELPER
# ============================================

def buat_folder(path_base):
    """Membuat struktur folder output"""
    folder_list = [
        f"{path_base}/images/mentah",
        f"{path_base}/images/setengah_matang",
        f"{path_base}/images/matang",
        f"{path_base}/labels/mentah",
        f"{path_base}/labels/setengah_matang",
        f"{path_base}/labels/matang",
    ]
    
    for folder in folder_list:
        Path(folder).mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Struktur folder berhasil dibuat di: {path_base}")

def salin_gambar(path_sumber, path_output, kelas_mapping):
    """Menyalin gambar ke folder terorganisir"""
    print("\n📂 Menyalin gambar...")
    
    for nama_folder, id_kelas in kelas_mapping.items():
        folder_sumber = os.path.join(path_sumber, nama_folder)
        folder_tujuan = os.path.join(path_output, "images", nama_folder)
        
        if not os.path.exists(folder_sumber):
            print(f"⚠️  Folder tidak ditemukan: {folder_sumber}")
            continue
        
        file_gambar = [f for f in os.listdir(folder_sumber) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for nama_file in tqdm(file_gambar, desc=f"Menyalin {nama_folder}"):
            sumber = os.path.join(folder_sumber, nama_file)
            tujuan = os.path.join(folder_tujuan, nama_file)
            shutil.copy2(sumber, tujuan)

def deteksi_bbox_buah(path_gambar, id_kelas):
    """
    Mendeteksi bounding box menggunakan color-based segmentation
    Berbeda untuk setiap tingkat kematangan
    """
    gambar = cv2.imread(path_gambar)
    if gambar is None:
        return []
    
    hsv = cv2.cvtColor(gambar, cv2.COLOR_BGR2HSV)
    h, w = gambar.shape[:2]
    
    # Definisi range warna per tingkat kematangan
    if id_kelas == 0:  # Mentah (hijau)
        lower = np.array([35, 40, 40])
        upper = np.array([90, 255, 255])
    elif id_kelas == 1:  # Setengah matang (kuning-orange)
        lower = np.array([15, 40, 40])
        upper = np.array([35, 255, 255])
    else:  # Matang (orange-merah)
        lower = np.array([5, 40, 40])
        upper = np.array([15, 255, 255])
    
    # Buat mask
    mask = cv2.inRange(hsv, lower, upper)
    
    # Operasi morfologi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Cari kontur
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # Filter noise
            continue
        
        x, y, bw, bh = cv2.boundingRect(contour)
        
        # Normalisasi koordinat ke 0-1
        x_center = (x + bw/2) / w
        y_center = (y + bh/2) / h
        width = bw / w
        height = bh / h
        
        # Validasi koordinat
        if 0 < x_center < 1 and 0 < y_center < 1 and 0 < width < 1 and 0 < height < 1:
            boxes.append({
                'class': id_kelas,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
    
    return boxes

def simpan_label_yolo(boxes, path_label):
    """Menyimpan boxes ke format YOLO (.txt)"""
    with open(path_label, 'w') as f:
        for box in boxes:
            line = f"{box['class']} {box['x_center']:.6f} {box['y_center']:.6f} {box['width']:.6f} {box['height']:.6f}\n"
            f.write(line)

def auto_label_gambar(path_sumber, path_output, kelas_mapping):
    """Auto-label semua gambar"""
    print("\n🏷️  Auto-labeling gambar...")
    
    for nama_folder, id_kelas in kelas_mapping.items():
        folder_gambar = os.path.join(path_output, "images", nama_folder)
        folder_label = os.path.join(path_output, "labels", nama_folder)
        
        file_gambar = [f for f in os.listdir(folder_gambar) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for nama_file in tqdm(file_gambar, desc=f"Labeling {nama_folder}"):
            path_gambar = os.path.join(folder_gambar, nama_file)
            nama_label = os.path.splitext(nama_file)[0] + '.txt'
            path_label = os.path.join(folder_label, nama_label)
            
            # Deteksi boxes
            boxes = deteksi_bbox_buah(path_gambar, id_kelas)
            
            # Simpan labels
            simpan_label_yolo(boxes, path_label)

def buat_data_yaml(path_output):
    """Membuat file data.yaml untuk YOLO"""
    isi_yaml = """path: dataset_organized
train: images
val: images
test: images

nc: 3
names: ['mentah', 'setengah_matang', 'matang']
"""
    
    path_yaml = os.path.join(path_output, "data.yaml")
    with open(path_yaml, 'w') as f:
        f.write(isi_yaml)
    
    print(f"\n✅ File dibuat: {path_yaml}")

def verifikasi_dataset(path_output, kelas_mapping):
    """Verifikasi dataset"""
    print("\n📊 Verifikasi Dataset:")
    print("-" * 50)
    
    total_gambar = 0
    total_label = 0
    
    for nama_folder in kelas_mapping.keys():
        folder_gambar = os.path.join(path_output, "images", nama_folder)
        folder_label = os.path.join(path_output, "labels", nama_folder)
        
        jumlah_gambar = len([f for f in os.listdir(folder_gambar) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        jumlah_label = len([f for f in os.listdir(folder_label) 
                           if f.endswith('.txt')])
        
        total_gambar += jumlah_gambar
        total_label += jumlah_label
        
        status = "✅" if jumlah_gambar == jumlah_label else "⚠️"
        print(f"{status} {nama_folder}: {jumlah_gambar} gambar, {jumlah_label} label")
    
    print("-" * 50)
    print(f"Total: {total_gambar} gambar, {total_label} label")
    
    if total_gambar == total_label == 300:
        print("✅ Dataset siap untuk training!")
    else:
        print("⚠️  Ada yang kurang atau berlebih!")

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("ORGANIZE & AUTO-LABEL DATASET KERSEN")
    print("=" * 50)
    
    # Step 1: Buat folder
    buat_folder(OUTPUT_PATH)
    
    # Step 2: Salin gambar
    salin_gambar(DATASETS_PATH, OUTPUT_PATH, KELAS_MAPPING)
    
    # Step 3: Auto-label
    auto_label_gambar(DATASETS_PATH, OUTPUT_PATH, KELAS_MAPPING)
    
    # Step 4: Buat data.yaml
    buat_data_yaml(OUTPUT_PATH)
    
    # Step 5: Verifikasi
    verifikasi_dataset(OUTPUT_PATH, KELAS_MAPPING)
    
    print("\n✨ Selesai! Dataset terorganisir dan auto-labeled!")