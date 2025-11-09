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

DATASETS_PATH = r"datasets"
OUTPUT_PATH = r"dataset_organized"
ANNOTATED_OUTPUT_PATH = r"dataset_annotated"

KELAS_MAPPING = {
    "mentah": 0,
    "setengah_matang": 1,
    "matang": 2
}

# ============================================
# FUNGSI HELPER
# ============================================

def hapus_folder_jika_ada(folder_paths):
    """Menghapus folder jika sudah ada untuk memastikan kebersihan."""
    for folder_path in folder_paths:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"🗑️  Folder lama dihapus: {folder_path}")

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

def buat_folder_anotasi(path_base):
    """Membuat struktur folder untuk output anotasi"""
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
    
    print(f"✅ Struktur folder anotasi berhasil dibuat di: {path_base}")

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
    Mendeteksi bounding box dengan range HSV yang luas
    """
    gambar = cv2.imread(path_gambar)
    if gambar is None:
        return []
    
    hsv = cv2.cvtColor(gambar, cv2.COLOR_BGR2HSV)
    h, w = gambar.shape[:2]
    
    # Range HSV yang lebih luas untuk deteksi lebih akurat
    if id_kelas == 0:  # Mentah (hijau)
        lower = np.array([25, 40, 40])
        upper = np.array([100, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif id_kelas == 1:  # Setengah matang (kuning-orange)
        lower = np.array([5, 40, 40])
        upper = np.array([45, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    else:  # Matang (merah) - Gabungkan dua rentang untuk merah
        # Rentang pertama (merah-pink)
        lower1 = np.array([0, 70, 50])
        upper1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        
        # Rentang kedua (merah tua)
        lower2 = np.array([170, 70, 50])
        upper2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower2, upper2)
        
        # Gabungkan mask
        mask = cv2.add(mask1, mask2)
    
    # Operasi morfologi yang lebih baik
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Cari kontur
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    # Iterasi melalui semua kontur yang ditemukan
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Abaikan kontur yang terlalu kecil (noise)
        if area < 500:
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

def gambar_dan_simpan_anotasi(path_gambar, path_label, path_output_anotasi, kelas_mapping):
    """Menggambar bounding box pada citra dan menyimpannya."""
    gambar = cv2.imread(path_gambar)
    if gambar is None:
        return

    h, w, _ = gambar.shape
    
    # Warna untuk setiap kelas (BGR)
    colors = {
        0: (0, 255, 0),   # Hijau untuk mentah
        1: (0, 255, 255), # Kuning untuk setengah_matang
        2: (0, 0, 255)    # Merah untuk matang
    }
    
    # Balikkan kelas_mapping untuk mendapatkan nama dari ID
    id_ke_nama_kelas = {v: k for k, v in kelas_mapping.items()}

    if os.path.exists(path_label):
        with open(path_label, 'r') as f:
            for line in f:
                parts = line.strip().split()
                id_kelas = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Konversi kembali ke koordinat piksel
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)

                # Gambar bounding box
                color = colors.get(id_kelas, (255, 255, 255)) # Default putih
                cv2.rectangle(gambar, (x1, y1), (x2, y2), color, 2)

                # Tambahkan label kelas
                nama_kelas = id_ke_nama_kelas.get(id_kelas, "Tidak Diketahui")
                cv2.putText(gambar, nama_kelas, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite(path_output_anotasi, gambar)

def simpan_gambar_beranotasi(path_output, path_anotasi, kelas_mapping):
    """Menyimpan semua gambar dengan anotasi bounding box."""
    print("\n🎨 Menggambar dan menyimpan anotasi...")
    
    for nama_folder in kelas_mapping.keys():
        folder_gambar = os.path.join(path_output, "images", nama_folder)
        folder_label = os.path.join(path_output, "labels", nama_folder)
        folder_output_anotasi = os.path.join(path_anotasi, "images", nama_folder)
        
        file_gambar = [f for f in os.listdir(folder_gambar) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for nama_file in tqdm(file_gambar, desc=f"Anotasi {nama_folder}"):
            path_gambar = os.path.join(folder_gambar, nama_file)
            nama_label = os.path.splitext(nama_file)[0] + '.txt'
            path_label = os.path.join(folder_label, nama_label)
            path_output_file = os.path.join(folder_output_anotasi, nama_file)
            
            gambar_dan_simpan_anotasi(path_gambar, path_label, path_output_file, kelas_mapping)

def salin_label_anotasi(path_sumber_label, path_output_anotasi, kelas_mapping):
    """Menyalin file label ke folder anotasi."""
    print("\n📋 Menyalin file label ke folder anotasi...")
    
    for nama_folder in kelas_mapping.keys():
        folder_sumber = os.path.join(path_sumber_label, "labels", nama_folder)
        folder_tujuan = os.path.join(path_output_anotasi, "labels", nama_folder)
        
        if not os.path.exists(folder_sumber):
            print(f"⚠️  Folder label sumber tidak ditemukan: {folder_sumber}")
            continue
        
        file_label = [f for f in os.listdir(folder_sumber) if f.endswith('.txt')]
        
        for nama_file in tqdm(file_label, desc=f"Menyalin label {nama_folder}"):
            sumber = os.path.join(folder_sumber, nama_file)
            tujuan = os.path.join(folder_tujuan, nama_file)
            shutil.copy2(sumber, tujuan)

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
    
    if total_gambar == total_label:
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
    
    # Hapus folder output lama jika ada
    hapus_folder_jika_ada([OUTPUT_PATH, ANNOTATED_OUTPUT_PATH])
    
    buat_folder(OUTPUT_PATH)
    buat_folder_anotasi(ANNOTATED_OUTPUT_PATH)
    salin_gambar(DATASETS_PATH, OUTPUT_PATH, KELAS_MAPPING)
    auto_label_gambar(DATASETS_PATH, OUTPUT_PATH, KELAS_MAPPING)
    simpan_gambar_beranotasi(OUTPUT_PATH, ANNOTATED_OUTPUT_PATH, KELAS_MAPPING)
    salin_label_anotasi(OUTPUT_PATH, ANNOTATED_OUTPUT_PATH, KELAS_MAPPING)
    buat_data_yaml(OUTPUT_PATH)
    verifikasi_dataset(OUTPUT_PATH, KELAS_MAPPING)
    
    print("\n✨ Selesai! Dataset terorganisir, auto-labeled, dan gambar anotasi disimpan!")