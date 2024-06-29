import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Load model untuk pengenalan ekspresi wajah
model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5')

# Inisialisasi modul OpenCV untuk deteksi wajah
mp_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inisialisasi video capture
cap = cv2.VideoCapture(0)  # Ganti ke 1 jika menggunakan kamera eksternal

while cap.isOpened():
    # Baca frame dari video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi ke grayscale untuk deteksi wajah
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah menggunakan Haar Cascade
    faces = mp_face.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Potong wajah dari frame dan ubah ukuran menjadi 64x64
        face_roi = cv2.resize(gray_frame[y:y+h, x:x+w], (64, 64))
        face_roi = np.expand_dims(np.expand_dims(face_roi, axis=-1), axis=0)

        # Normalisasi dan prediksi ekspresi wajah
        face_roi = face_roi / 255.0
        prediction = model.predict(face_roi)
        max_index = np.argmax(prediction[0])

        # Tentukan suasana hati berdasarkan indeks prediksi (dalam bahasa Indonesia)
        expressions = ["Marah", "Jijik", "Takut", "Senang", "Sedih", "Terkejut", "Netral"]
        expression_text = expressions[max_index]

        # Ambil persentase prediksi
        confidence = int(np.max(prediction[0]) * 100)

        # Gambar kotak dan teks hasil di frame (menggunakan warna hijau)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{expression_text} ({confidence}%)"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Tampilkan frame
    cv2.imshow("Deteksi Ekspresi Wajah", frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup video capture dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()

