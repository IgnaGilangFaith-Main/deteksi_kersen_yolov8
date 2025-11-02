"""
Script untuk split dataset menjadi train/val/test
Train: 240 (80%)
Val: 30 (10%)
Test: 30 (10%)
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================
# KONFIGURASI
# ============================================

PATH_SUMBER = r"dataset_organized"
PATH_OUTPUT = r"train_split"

KELAS_MAPPING = {
    "mentah": 0,
    "setengah_matang": 1,
    "matang": 2
}

# ============================================
# FUNGSI
# ============================================

def buat_folder_split(path_output):
    """Membuat struktur folder train/val/test"""
    splits = ['train', 'val', 'test']
    
    for split in splits:
        for subdir in ['images', 'labels']:
            path = os.path.join(path_output, subdir, split)
            Path(path).mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Struktur folder split berhasil dibuat")

def ambil_semua_file(path_sumber, kelas_mapping):
    """Mengambil semua file per class"""
    file_dict = {}
    
    for nama_kelas in kelas_mapping.keys():
        folder_gambar = os.path.join(path_sumber, "images", nama_kelas)
        
        file_list = [f for f in os.listdir(folder_gambar) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        file_dict[nama_kelas] = file_list
    
    return file_dict

def split_dan_salin(path_sumber, path_output, file_dict):
    """Split file dan salin ke train/val/test"""
    print("\n📊 Split dataset (80% train, 10% val, 10% test)...")
    
    for nama_kelas, file_list in file_dict.items():
        # Split: 80-10-10
        file_train, file_temp = train_test_split(file_list, test_size=0.2, random_state=42)
        file_val, file_test = train_test_split(file_temp, test_size=0.5, random_state=42)
        
        splits = {
            'train': file_train,
            'val': file_val,
            'test': file_test
        }
        
        print(f"\n{nama_kelas}:")
        print(f"  Train: {len(file_train)}")
        print(f"  Val: {len(file_val)}")
        print(f"  Test: {len(file_test)}")
        
        # Salin file
        for nama_split, daftar_file in splits.items():
            for nama_file in tqdm(daftar_file, desc=f"  Menyalin ke {nama_split}"):
                # Salin gambar
                src_gambar = os.path.join(path_sumber, "images", nama_kelas, nama_file)
                dst_gambar = os.path.join(path_output, "images", nama_split, nama_file)
                shutil.copy2(src_gambar, dst_gambar)
                
                # Salin label
                nama_label = os.path.splitext(nama_file)[0] + '.txt'
                src_label = os.path.join(path_sumber, "labels", nama_kelas, nama_label)
                dst_label = os.path.join(path_output, "labels", nama_split, nama_label)
                shutil.copy2(src_label, dst_label)

def buat_data_yaml(path_output):
    """Membuat data.yaml untuk training"""
    isi_yaml = """path: train_split
train: images/train
val: images/val
test: images/test

nc: 3
names: ['mentah', 'setengah_matang', 'matang']
"""
    
    path_yaml = os.path.join(path_output, "data.yaml")
    with open(path_yaml, 'w') as f:
        f.write(isi_yaml)
    
    print(f"\n✅ File dibuat: {path_yaml}")

def verifikasi_split(path_output):
    """Verifikasi split"""
    print("\n✅ Verifikasi Split Dataset:")
    print("-" * 50)
    
    splits = ['train', 'val', 'test']
    total = 0
    
    for split in splits:
        jumlah_gambar = len([f for f in os.listdir(os.path.join(path_output, "images", split)) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        jumlah_label = len([f for f in os.listdir(os.path.join(path_output, "labels", split)) 
                           if f.endswith('.txt')])
        
        total += jumlah_gambar
        status = "✅" if jumlah_gambar == jumlah_label else "⚠️"
        print(f"{status} {split.upper()}: {jumlah_gambar} gambar")
    
    print("-" * 50)
    print(f"Total: {total} gambar")

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("SPLIT DATASET (Train/Val/Test)")
    print("=" * 50)
    
    # Step 1: Buat folder
    buat_folder_split(PATH_OUTPUT)
    
    # Step 2: Ambil semua file
    file_dict = ambil_semua_file(PATH_SUMBER, KELAS_MAPPING)
    
    # Step 3: Split dan salin
    split_dan_salin(PATH_SUMBER, PATH_OUTPUT, file_dict)
    
    # Step 4: Buat data.yaml
    buat_data_yaml(PATH_OUTPUT)
    
    # Step 5: Verifikasi
    verifikasi_split(PATH_OUTPUT)
    
    print("\n✨ Split dataset selesai!")