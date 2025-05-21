# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ===============================
# Load model and data
# ===============================
model = joblib.load('naive_bayes_model.pkl')  # Pastikan file ini tersedia
iris = load_iris()

# Buat DataFrame dari data iris
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
target_names = iris.target_names

# ===============================
# Sidebar Navigation
# ===============================
st.sidebar.title("🌼 Iris Classifier App")
page = st.sidebar.radio("Pilih Halaman", ["Deskripsi Data", "Prediksi", "Visualisasi"])

# ===============================
# Page 1: Deskripsi Data
# ===============================
if page == "Deskripsi Data":
    st.title("🔍 Deskripsi Dataset Iris")
    st.markdown("""
    Dataset Iris memiliki 150 data bunga dengan 4 fitur:
    - Panjang sepal
    - Lebar sepal
    - Panjang petal
    - Lebar petal
    
    Terdapat 3 jenis bunga:
    - Setosa
    - Versicolor
    - Virginica
    """)

    st.subheader("Contoh Data")
    st.dataframe(df.head())

    st.subheader("Statistik Ringkasan")
    st.write(df.describe())

# ===============================
# Page 2: Prediksi
# ===============================
elif page == "Prediksi":
    st.title("🌸 Prediksi Jenis Bunga Iris")
    st.write("Masukkan panjang dan lebar sepal & petal bunga:")

    # Input pengguna
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

    if st.button("Prediksi"):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)[0]
        predicted_class = target_names[prediction]
        st.success(f"Prediksi: **{predicted_class}** 🌼")

# ===============================
# Page 3: Visualisasi
# ===============================
elif page == "Visualisasi":
    st.title("📊 Visualisasi Dataset Iris")

    st.subheader("Pairplot Berdasarkan Jenis")
    df_plot = df.copy()
    df_plot['species'] = df_plot['target'].apply(lambda x: target_names[x])
    sns.set(style="whitegrid")
    fig = sns.pairplot(df_plot, hue="species", diag_kind="kde")
    st.pyplot(fig)
