"""
Flask Backend untuk Deteksi Kematangan Kersen
Real-time Sync Version (BGR color fixed)
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
import logging

# ============================================
# LOGGING SETUP
# ============================================

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ============================================
# INISIALISASI FLASK
# ============================================

app = Flask(__name__)
CORS(app)

# ============================================
# LOAD MODEL
# ============================================

print("\n" + "=" * 60)
print("DETEKSI KEMATANGAN KERSEN - WEB APP")
print("=" * 60 + "\n")

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
model = None

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model tidak ditemukan: {MODEL_PATH}")
    print("⚠️  Jalankan script training terlebih dahulu!")
else:
    try:
        model = YOLO(MODEL_PATH)
        print(f"✅ Model berhasil dimuat: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        logger.error(f"Model loading error: {e}")

print()

# Nama kelas
NAMA_KELAS = {
    0: "Mentah",
    1: "Setengah Matang",
    2: "Matang"
}

# PENTING: BGR format for cv2.rectangle
WARNA_KELAS = {
    0: (0, 255, 0),        # Mentah (Hijau)
    1: (0, 255, 255),      # Setengah Matang (Kuning)
    2: (0, 0, 255)         # Matang (Merah)
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
    
    logger.debug("Menerima request deteksi...")
    
    if model is None:
        logger.error("Model belum dimuat!")
        return jsonify({
            'status': 'error',
            'message': 'Model belum dimuat. Jalankan training terlebih dahulu!'
        }), 500
    
    try:
        # Ambil gambar dan confidence dari request
        data = request.get_json()
        if not data or 'image' not in data:
            logger.error("Image tidak ditemukan di request")
            return jsonify({
                'status': 'error', 
                'message': 'Gambar tidak ditemukan'
            }), 400
        
        logger.debug("Decoding base64 image...")
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # Convert PIL image to numpy array
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        logger.debug(f"Frame shape: {frame.shape}")
        
        # Ambil confidence dari client (atau gunakan default)
        confidence = data.get('confidence', 0.3)
        
        # Deteksi dengan confidence dari client
        logger.debug(f"Menjalankan tracking dengan confidence {confidence}...")
        hasil = model.track(frame, conf=confidence, verbose=False, device=device, persist=True)
        
        # Draw bounding box
        frame_hasil = frame.copy()
        deteksi_list = []
        
        logger.debug(f"Jumlah deteksi: {len(hasil[0].boxes)}")
        
        # Pastikan ada hasil tracking sebelum diproses
        if hasil[0].boxes.id is not None:
            boxes = hasil[0].boxes.xyxy.cpu()
            clss = hasil[0].boxes.cls.cpu().tolist()
            confs = hasil[0].boxes.conf.cpu().tolist()
            track_ids = hasil[0].boxes.id.int().cpu().tolist()

            for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
                # Get coordinates
                x1, y1, x2, y2 = map(int, box)
                id_kelas = int(cls)
                confidence_score = float(conf)
                nama_kelas = NAMA_KELAS.get(id_kelas, "Unknown")
                warna = WARNA_KELAS.get(id_kelas, (255, 255, 255))
                
                # Draw rectangle
                cv2.rectangle(frame_hasil, (x1, y1), (x2, y2), warna, 2)
                
                # Put text with Track ID
                label = f"ID:{track_id} {nama_kelas} ({confidence_score:.2f})"
                cv2.putText(frame_hasil, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, warna, 2)
                
                # Tambah ke list
                deteksi_list.append({
                    'track_id': track_id,
                    'kelas': nama_kelas,
                    'confidence': round(confidence_score, 2),
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })
                
                logger.debug(f"Track ID: {track_id} - Deteksi: {nama_kelas} ({confidence_score:.2f})")
        
        # Convert kembali ke base64
        _, buffer = cv2.imencode('.jpg', frame_hasil)
        image_base64 = base64.b64encode(buffer).decode()
        
        logger.debug("Response berhasil dibuat")
        
        return jsonify({
            'status': 'success',
            'image': f'data:image/jpeg;base64,{image_base64}',
            'deteksi': deteksi_list,
            'jumlah_deteksi': len(deteksi_list)
        })
    
    except Exception as e:
        logger.error(f"Error deteksi: {e}", exc_info=True)
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
    model_status = model is not None
    
    logger.debug(f"Health check - Model loaded: {model_status}")
    
    if not model_status:
        return jsonify({
            'status': 'error',
            'model_loaded': False,
            'message': 'Model belum dimuat'
        }), 500
    
    return jsonify({
        'status': 'ok',
        'model_loaded': True,
        'classes': NAMA_KELAS,
        'device': str(device)
    })

# ============================================
# ERROR HANDLER
# ============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Route tidak ditemukan'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Server error: {error}")
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("📡 Server berjalan di: http://localhost:5000")
    print("📹 Buka browser dan akses: http://localhost:5000")
    print("✨ Tekan Ctrl+C untuk menghentikan server")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)