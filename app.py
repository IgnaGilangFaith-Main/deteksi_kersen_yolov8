"""
Flask Backend untuk Deteksi Kematangan Kersen
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
import torch
from PIL import Image
from io import BytesIO

# ============================================
# INISIALISASI FLASK
# ============================================

app = Flask(__name__)
CORS(app)

# ============================================
# LOAD MODEL
# ============================================

print("\n" + "=" * 50)
print("DETEKSI KEMATANGAN KERSEN - WEB APP")
print("=" * 50 + "\n")

# Cek device
gpu_tersedia = torch.cuda.is_available()

print(f"GPU Tersedia: {gpu_tersedia}")
if gpu_tersedia:
    device = 0
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print(f"Device: CPU")

print()

# Load model
MODEL_PATH = r"models/yolov8n_kersen_best.pt"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model tidak ditemukan: {MODEL_PATH}")
    print("⚠️  Jalankan script training terlebih dahulu!")
    model = None
else:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model berhasil dimuat: {MODEL_PATH}")

print()

# Nama kelas
NAMA_KELAS = {
    0: "Mentah",
    1: "Setengah Matang",
    2: "Matang"
}

WARNA_KELAS = {
    0: (0, 255, 0),
    1: (0, 165, 255),
    2: (0, 0, 255)
}

# ============================================
# ROUTE - HOMEPAGE
# ============================================

@app.route('/')
def index():
    """Halaman utama"""
    return render_template('index.html')

# ============================================
# ROUTE - DETEKSI GAMBAR
# ============================================

@app.route('/detect', methods=['POST'])
def detect():
    """API untuk deteksi dari gambar"""
    
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model belum dimuat. Jalankan training terlebih dahulu!'
        }), 500
    
    try:
        # Ambil gambar dari request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'status': 'error', 'message': 'Gambar tidak ditemukan'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert PIL image to numpy array
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Deteksi
        hasil = model.predict(frame, conf=0.5, verbose=False, device=device)
        
        # Draw bounding box
        frame_hasil = frame.copy()
        deteksi_list = []
        
        for box in hasil[0].boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            id_kelas = int(box.cls[0])
            confidence = float(box.conf[0])
            nama_kelas = NAMA_KELAS.get(id_kelas, "Unknown")
            warna = WARNA_KELAS.get(id_kelas, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(frame_hasil, (x1, y1), (x2, y2), warna, 2)
            
            # Put text
            label = f"{nama_kelas} ({confidence:.2f})"
            cv2.putText(frame_hasil, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, warna, 2)
            
            # Tambah ke list
            deteksi_list.append({
                'kelas': nama_kelas,
                'confidence': round(confidence, 2),
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            })
        
        # Convert kembali ke base64
        _, buffer = cv2.imencode('.jpg', frame_hasil)
        image_base64 = base64.b64encode(buffer).decode()
        
        return jsonify({
            'status': 'success',
            'image': f'data:image/jpeg;base64,{image_base64}',
            'deteksi': deteksi_list,
            'jumlah_deteksi': len(deteksi_list)
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}'
        }), 500

# ============================================
# ROUTE - HEALTH CHECK
# ============================================

@app.route('/health', methods=['GET'])
def health():
    """Check status model"""
    if model is None:
        return jsonify({
            'status': 'error',
            'model_loaded': False,
            'message': 'Model belum dimuat'
        }), 500
    
    return jsonify({
        'status': 'ok',
        'model_loaded': True,
        'classes': NAMA_KELAS
    })

# ============================================
# ERROR HANDLER
# ============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Route tidak ditemukan'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("📡 Server berjalan di: http://localhost:5000")
    print("📹 Buka browser dan akses: http://localhost:5000")
    print("✨ Tekan Ctrl+C untuk menghentikan server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)