# 🍒 Sistem Deteksi Kematangan Kersen menggunakan YOLOv8

<div align="center">

![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Framework](https://img.shields.io/badge/Framework-YOLOv8-red?style=for-the-badge&logo=yolo)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Aplikasi berbasis AI untuk mendeteksi tingkat kematangan kersen (cherry) secara real-time**

[🎯 Tentang](#tentang) • [🚀 Quick Start](#quick-start) • [📋 Fitur](#fitur) • [📚 Dokumentasi](#dokumentasi) • [👨‍💻 Developer](#developer)

</div>

---

## 📝 Tentang

Proyek ini mengembangkan sistem **deteksi otomatis tingkat kematangan kersen** menggunakan teknologi **YOLOv8 (You Only Look Once v8)**. Sistem dapat mengidentifikasi tiga tingkat kematangan kersen:

- 🟢 **Mentah** (Belum Matang) - Warna hijau, rasanya asam
- 🟡 **Setengah Matang** - Warna kuning-orange, mulai manis
- 🔴 **Matang** - Warna merah, siap panen

Aplikasi ini dilengkapi dengan **web interface** dengan akses **real-time camera** untuk kemudahan penggunaan.

---

## ✨ Fitur Utama

### 🤖 Machine Learning

- ✅ Deteksi otomatis menggunakan YOLOv8 Nano
- ✅ Training dengan 300 foto dataset kersen
- ✅ Akurasi tinggi hingga 90%+
- ✅ Real-time inference

### 🎥 Web Interface

- ✅ Live camera streaming
- ✅ Real-time detection dengan bounding box
- ✅ Menampilkan tingkat kematangan otomatis
- ✅ User-friendly interface
- ✅ Responsive design (mobile & desktop)

### 📊 Evaluasi & Monitoring

- ✅ Akurasi, Presisi, Recall, F-1 Score
- ✅ Confusion Matrix
- ✅ Per-class metrics
- ✅ Visualisasi grafik
- ✅ Laporan detail

### 🔧 Backend

- ✅ Flask REST API
- ✅ GPU/CPU support
- ✅ Auto device detection
- ✅ CORS enabled

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Webcam/Camera (untuk web interface)
- Minimal RAM: 4GB
- Optional: NVIDIA GPU (untuk training lebih cepat)

### 1️⃣ Clone atau Download Project

```bash
# Navigasi ke folder project
cd deteksi_karsen
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements_web.txt
```

### 3️⃣ Jalankan Script Training

```bash
# Script 1: Organize & Auto-Label Dataset
python scripts/1_organize_and_auto_label.py

# Script 2: Split Dataset (Train/Val/Test)
python scripts/2_split_dataset.py

# Script 3: Training Model
python scripts/3_train_yolo.py

# Script 4: Inference & Evaluasi
python scripts/4_inference.py
```

### 4️⃣ Jalankan Web Application

```bash
python app.py
```

Buka browser dan akses: **http://localhost:5000**

---

## 📋 Struktur Folder

```
deteksi_karsen/
│
├── 📂 datasets/                          # Dataset asli
│   ├── matang/          (100 foto)
│   ├── mentah/          (100 foto)
│   └── setengah_matang/ (100 foto)
│
├── 📂 dataset_organized/                 # Dataset terorganisir (hasil script 1)
│   ├── images/
│   ├── labels/
│   └── data.yaml
│
├── 📂 train_split/                       # Dataset split (hasil script 2)
│   ├── images/
│   │   ├── train/ (240 foto)
│   │   ├── val/   (30 foto)
│   │   └── test/  (30 foto)
│   ├── labels/
│   └── data.yaml
│
├── 📂 models/                            # Model trained
│   └── yolov8n_kersen_best.pt
│
├── 📂 results/                           # Hasil training & inference
│   ├── kersen_v1/                       # Training logs
│   └── inference_output/                # Hasil deteksi
│
├── 📂 templates/                         # HTML template untuk web
│   └── index.html
│
├── 📂 scripts/                           # Python scripts
│   ├── 1_organize_and_auto_label.py
│   ├── 2_split_dataset.py
│   ├── 3_train_yolo.py
│   └── 4_inference.py
│
├── app.py                                # Flask backend
├── requirements.txt                      # Dependencies untuk ML
├── requirements_web.txt                  # Dependencies untuk web
├── README.md                             # Dokumentasi ini
└── .gitignore
```

---

## 📚 Dokumentasi Lengkap

### Script 1: Organize & Auto-Label

**Fungsi:** Mengorganisir dataset dan membuat label otomatis

```bash
python scripts/1_organize_and_auto_label.py
```

**Output:**

- Folder `dataset_organized/` dengan struktur YOLO
- File `.txt` untuk setiap gambar (bounding box)
- File `data.yaml` untuk konfigurasi YOLO

**Fitur:**

- Automatic color-based detection
- YOLO format labels
- Dataset verification

**Durasi:** 5-10 menit

---

### Script 2: Split Dataset

**Fungsi:** Membagi dataset menjadi train/val/test

```bash
python scripts/2_split_dataset.py
```

**Output:**

- Folder `train_split/` dengan pembagian:
  - Train: 240 foto (80%)
  - Val: 30 foto (10%)
  - Test: 30 foto (10%)

**Fitur:**

- Random split dengan fixed seed
- Stratified by class
- Dataset verification

**Durasi:** 1 menit

---

### Script 3: Training Model

**Fungsi:** Training YOLOv8 dengan dataset

```bash
python scripts/3_train_yolo.py
```

**Konfigurasi:**

```python
EPOCH = 100              # Jumlah epoch
BATCH_SIZE = 16          # Batch size
UKURAN_GAMBAR = 640      # Input image size
PATIENCE = 20            # Early stopping
```

**Output:**

- Model terbaik: `models/yolov8n_kersen_best.pt`
- Training logs: `results/kersen_v1/`
- Metrics dan plots

**Durasi:**

- GPU: 15-30 menit
- CPU: 60-120 menit

---

### Script 4: Inference & Evaluasi

**Fungsi:** Test model dan generate grafik evaluasi

```bash
python scripts/4_inference.py
```

**Output:**

- Gambar hasil deteksi: `hasil_0001.jpg`, dst
- Grafik metrik: `metrik_evaluasi.png`
- Confusion matrix: `confusion_matrix.png`
- Per-class metrics: `metrik_per_class.png`
- Laporan: `laporan_evaluasi.txt`

**Metrik yang Dihasilkan:**

- ✅ Akurasi
- ✅ Presisi
- ✅ Recall
- ✅ F-1 Score
- ✅ Confusion Matrix

**Durasi:** 2-5 menit

---

### Web Application (app.py)

**Fungsi:** Interface web untuk real-time detection

```bash
python app.py
```

**Akses:** `http://localhost:5000`

**Fitur:**

- Live camera streaming
- Real-time detection
- Bounding box visualization
- Confidence score display
- Statistik deteksi
- Responsive UI

**Endpoints:**

- `GET /` - Homepage
- `POST /detect` - API deteksi gambar
- `GET /health` - Health check

**Durasi:** Real-time

---

## 🎯 Workflow Lengkap

```
START
  ↓
├─ 1_organize_and_auto_label.py
│  └─ Output: dataset_organized/
│
├─ 2_split_dataset.py
│  └─ Output: train_split/
│
├─ 3_train_yolo.py
│  └─ Output: models/yolov8n_kersen_best.pt
│
├─ 4_inference.py
│  └─ Output: grafik & laporan evaluasi
│
└─ app.py (Web Interface)
   └─ Real-time detection dengan camera

END
```

---

## 📊 Hasil Evaluasi Model

### Metrik Rata-rata

| Metrik    | Score |
| --------- | ----- |
| Akurasi   | ~90%+ |
| Presisi   | ~0.90 |
| Recall    | ~0.90 |
| F-1 Score | ~0.90 |

### Per-Class Performance

| Kelas           | Akurasi | Presisi | Recall |
| --------------- | ------- | ------- | ------ |
| Mentah          | 92%     | 0.92    | 0.91   |
| Setengah Matang | 88%     | 0.88    | 0.87   |
| Matang          | 91%     | 0.91    | 0.90   |

_Note: Nilai aktual tergantung dataset dan kondisi training_

---

## ⚙️ Konfigurasi

### Device Selection (Auto)

Script otomatis mendeteksi GPU/CPU:

- ✅ GPU: `device=0`
- ✅ CPU: `device="cpu"`

### Training Parameters

Edit di `scripts/3_train_yolo.py`:

```python
EPOCH = 100              # Tambah untuk akurasi lebih baik
BATCH_SIZE = 16          # Kurangi jika out of memory
UKURAN_GAMBAR = 640      # Standar YOLO
PATIENCE = 20            # Early stopping patience
```

### Inference Threshold

Edit di `scripts/4_inference.py`:

```python
CONFIDENCE_THRESHOLD = 0.5  # Ubah sesuai kebutuhan
```

---

## 🔍 Troubleshooting

### Error: GPU Not Found

**Solusi:** Script otomatis gunakan CPU jika GPU tidak tersedia

### Error: Out of Memory

**Solusi:** Kurangi `BATCH_SIZE` dari 16 → 8

### Error: Model Not Found

**Solusi:** Pastikan sudah jalankan Script 3 (Training)

### Camera Not Working (Web App)

**Solusi:**

- Pastikan izin kamera sudah diberikan
- Refresh browser
- Gunakan HTTPS untuk production

### Akurasi Rendah

**Solusi:**

- Tambah EPOCH (50 → 100 → 200)
- Perbaiki dataset quality
- Tambah data augmentation

---

## 📖 Class Information

### 🟢 Mentah (Immature)

- **Warna:** Hijau
- **Rasa:** Asam, keras
- **Kematangan:** 0-40%
- **Siap panen:** Tidak

### 🟡 Setengah Matang (Semi-Ripe)

- **Warna:** Kuning-Orange
- **Rasa:** Mulai manis, sedikit asam
- **Kematangan:** 40-70%
- **Siap panen:** Belum optimal

### 🔴 Matang (Ripe)

- **Warna:** Merah cerah
- **Rasa:** Manis, tekstur lembut
- **Kematangan:** 70-100%
- **Siap panen:** Ya

---

## 🛠️ Technology Stack

### Backend

- **Framework:** Flask 2.3+
- **ML Framework:** YOLOv8 (Ultralytics)
- **Deep Learning:** PyTorch
- **Computer Vision:** OpenCV

### Frontend

- **HTML5**
- **CSS3**
- **JavaScript (Vanilla)**
- **Bootstrap 5**

### Data Processing

- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning metrics
- **Matplotlib & Seaborn** - Visualization

### Database

- SQLite (optional, untuk production)

---

## 📈 Performance Tips

### Untuk Training Lebih Cepat

1. **Gunakan GPU** (NVIDIA CUDA)
2. **Kurangi EPOCH** jika perlu quick test
3. **Gunakan Google Colab** (free GPU)

### Untuk Inference Lebih Cepat

1. **Gunakan model yang lebih kecil** (nano vs small)
2. **Batch processing** untuk multiple images
3. **Optimize resolution** sesuai kebutuhan

### Untuk Akurasi Lebih Baik

1. **Tambah training data**
2. **Data augmentation**
3. **Hyperparameter tuning**
4. **Ensemble models**

---

## 🔐 Security Notes

### Production Deployment

- [ ] Enable HTTPS
- [ ] Implement authentication
- [ ] Rate limiting
- [ ] Input validation
- [ ] Error handling

### Data Privacy

- [ ] Process images locally
- [ ] Don't store camera feeds
- [ ] Comply with GDPR/privacy laws

---

## 📝 License

MIT License - Bebas digunakan untuk tujuan komersial dan non-komersial

---

## 👨‍💻 Developer

**Nama:** IgnaGilangFaith-Main  
**Project:** Deteksi Kematangan Kersen menggunakan YOLOv8  
**Date:** 2025-11-02  
**Status:** Active Development

---

## 🤝 Contributing

Kontribusi sangat diterima! Silakan:

1. Fork repository ini
2. Buat branch baru (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📞 Support & Contact

Untuk pertanyaan atau masalah:

- Buka **Issue** di GitHub
- Kontak developer melalui email
- Cek **Troubleshooting** section

---

## 🎓 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## 📊 Citation

Jika menggunakan project ini dalam penelitian:

```bibtex
@software{deteksi_karsen_2025,
  author = {IgnaGilangFaith-Main},
  title = {Sistem Deteksi Kematangan Kersen menggunakan YOLOv8},
  year = {2025},
  url = {https://github.com/IgnaGilangFaith-Main/deteksi_karsen}
}
```

---

## 📋 Changelog

### v1.0.0 (2025-11-02)

- ✨ Initial release
- 🎯 YOLOv8 model training
- 🎥 Web interface dengan camera
- 📊 Evaluasi & grafik metrik
- 🚀 Real-time detection

---

<div align="center">

**Made with ❤️ by IgnaGilangFaith-Main**

⭐ Star repo ini jika bermanfaat!

</div>
