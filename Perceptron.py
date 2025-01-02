import streamlit as st
import pickle
import numpy as np

# Memuat model, scaler, dan label encoder yang sudah disimpan
def load_model_and_encoder():
    with open('fruit_perceptron.sav', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler_perceptron.sav', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('fruit_encoder.sav', 'rb') as encoder_file:
        label_encoder = pickle.load(encoder_file)
    return model, scaler, label_encoder
# Memuat model, scaler, dan label encoder
model, scaler, label_encoder = load_model_and_encoder()

# Judul Aplikasi
st.title("Aplikasi Prediksi Klasifikasi Buah")
st.title("Masukkan Fitur")
st.markdown("---")

# Input data oleh pengguna
diameter = st.number_input("Masukkan Diameter (cm):", min_value=0.0, format="%.2f")
weight = st.number_input("Masukkan Berat (gram):", min_value=0.0, format="%.2f")
red = st.number_input("Masukkan Nilai Merah (0-255):", min_value=0, max_value=255)
green = st.number_input("Masukkan Nilai Hijau (0-255):", min_value=0, max_value=255)
blue = st.number_input("Masukkan Nilai Biru (0-255):", min_value=0, max_value=255)

# Jika tombol prediksi ditekan
if st.button("Prediksi"):
    try:
        # Validasi input
        if not all([diameter, weight, red, green, blue]):
            st.warning("Semua input harus diisi.")
        else:
            # Mengubah data input menjadi array
            data_baru = np.array([[diameter, weight, red, green, blue]])
            
            # Normalisasi data menggunakan scaler yang sudah dilatih
            data_baru_scaled = scaler.transform(data_baru)
            
            # Prediksi menggunakan model
            prediksi = model.predict(data_baru_scaled)
            
            # Mengembalikan hasil prediksi ke label asli
            label = label_encoder.inverse_transform(prediksi)[0]
            
            # Menampilkan hasil prediksi
            st.success(f"Prediksi Klasifikasi: {label}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# Footer
st.markdown("---")
