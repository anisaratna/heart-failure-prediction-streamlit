# File: app.py (Versi Final dengan Visualisasi Plotly)

import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go # <-- Import library Plotly

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Prediksi Gagal Jantung", page_icon="❤️", layout="wide")

# --- FUNGSI UNTUK MEMUAT MODEL ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('heart_failure_model.joblib')
        return model
    except FileNotFoundError:
        st.error("File 'heart_failure_model.joblib' tidak ditemukan. Harap jalankan notebook training di Colab dan download modelnya.")
        return None

# Memuat model
model = load_model()

# --- TAMPILAN UI INPUT ---
if model is not None:
    st.title("❤️ Aplikasi Prediksi Gagal Jantung")
    st.write("Masukkan data klinis pasien untuk memprediksi risiko kematian akibat gagal jantung.")

    # Layout input menggunakan kolom agar lebih rapi
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Data Diri & Kebiasaan")
        age = st.number_input('Usia (Tahun)', min_value=1, max_value=120, value=60)
        sex = st.selectbox('Jenis Kelamin', options=[0, 1], format_func=lambda x: 'Perempuan' if x == 0 else 'Laki-laki')
        smoking = st.selectbox('Merokok', options=[0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
        
    with col2:
        st.subheader("Kondisi Medis")
        anaemia = st.selectbox('Anemia', options=[0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
        diabetes = st.selectbox('Diabetes', options=[0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
        high_blood_pressure = st.selectbox('Tekanan Darah Tinggi', options=[0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')

    with col3:
        st.subheader("Data Lab Klinis")
        creatinine_phosphokinase = st.number_input('Kreatinina Fosfokinase (mcg/L)', min_value=20, value=580)
        ejection_fraction = st.number_input('Fraksi Ejeksi (%)', min_value=10, max_value=80, value=38)
        platelets = st.number_input('Trombosit (kiloplatelets/mL)', min_value=25.0, value=263.0, step=1.0)
        serum_creatinine = st.number_input('Kreatinina Serum (mg/dL)', min_value=0.5, value=1.4, format="%.2f")
        serum_sodium = st.number_input('Natrium Serum (mEq/L)', min_value=110, value=137)
        time = st.number_input('Waktu Follow-up (Hari)', min_value=4, value=90)

    # --- Tombol Prediksi & Tampilan Hasil ---
    if st.button('**Buat Prediksi Sekarang**', use_container_width=True, type="primary"):
        
        # Kumpulkan input menjadi DataFrame
        feature_columns = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
        input_data = {
            'age': age, 'anaemia': anaemia, 'creatinine_phosphokinase': creatinine_phosphokinase, 'diabetes': diabetes,
            'ejection_fraction': ejection_fraction, 'high_blood_pressure': high_blood_pressure, 'platelets': platelets, 
            'serum_creatinine': serum_creatinine, 'serum_sodium': serum_sodium, 'sex': sex, 'smoking': smoking, 'time': time
        }
        input_df = pd.DataFrame([input_data])[feature_columns]
        
        # Lakukan prediksi
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Ekstrak probabilitas
        prob_selamat = prediction_proba[0][0]
        prob_kematian = prediction_proba[0][1]
        
        # Tentukan hasil utama dan tingkat keyakinan
        if prediction[0] == 0:
            hasil_prediksi = "SELAMAT"
            warna_hasil = "green"
            tingkat_keyakinan = prob_selamat
        else:
            hasil_prediksi = "BERISIKO TINGGI (FATAL)"
            warna_hasil = "red"
            tingkat_keyakinan = prob_kematian

        # =================================================================
        # --- VISUALISASI HASIL ---
        # =================================================================
        st.divider()
        st.header(f"Hasil Prediksi: Pasien Diprediksi **{hasil_prediksi}**")
        
        col_viz1, col_viz2 = st.columns(2)

        with col_viz1:
            # Gauge Chart untuk Tingkat Keyakinan
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = tingkat_keyakinan * 100,
                title = {'text': f"Tingkat Keyakinan (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': warna_hasil},
                    'steps' : [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"}],
                    'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 95}}))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_viz2:
            # Bar Chart untuk Distribusi Probabilitas
            fig_bar = go.Figure(go.Bar(
                x=[f"{prob_selamat*100:.1f}%", f"{prob_kematian*100:.1f}%"],
                y=['Selamat', 'Berisiko Tinggi'],
                orientation='h',
                marker_color=['green', 'red'],
                text=[f"{prob_selamat*100:.1f}%", f"{prob_kematian*100:.1f}%"],
                textposition='auto'))
            fig_bar.update_layout(title_text="Distribusi Probabilitas")
            st.plotly_chart(fig_bar, use_container_width=True)