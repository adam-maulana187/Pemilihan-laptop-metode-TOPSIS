import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# ------------------------------------------------------------------------------
# Page configuration
st.set_page_config(page_title="SPK Laptop - TOPSIS", layout="wide")
st.title("📊 Sistem Pendukung Keputusan Cerdas Pemilihan Laptop")
st.markdown("#### Metode **TOPSIS** (Technique for Order Preference by Similarity to Ideal Solution)")


# ------------------------------------------------------------------------------
# Sidebar: Upload & Parameters
st.sidebar.header("⚙️ Pengaturan")

# File uploader
uploaded_file = st.sidebar.file_uploader("Unggah dataset laptop (Excel)", type=["xlsx"])
default_file_path = "Cleaned_dataset.xlsx"  # we will use provided file, but in deployment we embed default

# Use uploaded file or default
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file, sheet_name="laptop_dataset")
    st.sidebar.success("Dataset berhasil dimuat!")
else:
    # For the purpose of this assignment, we need to load the provided file.
    # Since the file is given locally, we assume it's in the same directory.
    # In Streamlit Cloud, we can upload. We'll try to load default.
    try:
        data = pd.read_excel("Cleaned_dataset.xlsx", sheet_name="laptop_dataset")
        st.sidebar.info("Menggunakan dataset default (Cleaned_dataset.xlsx)")
    except:
        st.sidebar.error("File default tidak ditemukan. Silakan unggah file Excel.")
        st.stop()

# Preprocessing function
def preprocess_laptop_data(df):
    """
    Preprocess the raw dataset:
    - Keep relevant columns: Ram, HDD, SSD, Weight, Price
    - Convert Ram from string to numeric (e.g., '8GB' -> 8)
    - Convert Weight from string to numeric (e.g., '1.37kg' -> 1.37)
    - Compute Storage = HDD + SSD (both in GB)
    - Price: ensure numeric (values are in thousand Rupiah -> multiply by 1000)
    - Drop rows with missing critical values
    """
    df_clean = df.copy()
    # Ram: extract numeric part
    if 'Ram' in df_clean.columns:
        df_clean['RAM'] = df_clean['Ram'].astype(str).str.extract(r'(\d+)').astype(float)
    else:
        st.error("Kolom 'Ram' tidak ditemukan")
        return None

    # Storage: HDD + SSD
    if 'HDD' in df_clean.columns and 'SSD' in df_clean.columns:
        df_clean['HDD'] = pd.to_numeric(df_clean['HDD'], errors='coerce').fillna(0)
        df_clean['SSD'] = pd.to_numeric(df_clean['SSD'], errors='coerce').fillna(0)
        df_clean['Storage'] = df_clean['HDD'] + df_clean['SSD']
    else:
        st.error("Kolom HDD/SSD tidak ditemukan")
        return None

    # Weight: extract numeric
    if 'Weight' in df_clean.columns:
        df_clean['Berat'] = df_clean['Weight'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    else:
        st.error("Kolom 'Weight' tidak ditemukan")
        return None

    # Price: numeric, assume in thousand Rupiah -> convert to full Rupiah
    if 'Price' in df_clean.columns:
        df_clean['Harga'] = pd.to_numeric(df_clean['Price'], errors='coerce') * 1000
    else:
        st.error("Kolom 'Price' tidak ditemukan")
        return None

    # Create a descriptive name column: Company + TypeName + (key specs)
    if 'Company' in df_clean.columns and 'TypeName' in df_clean.columns:
        df_clean['Model'] = df_clean['Company'] + " " + df_clean['TypeName']
    else:
        df_clean['Model'] = "Laptop " + df_clean.index.astype(str)

    # Select final columns for TOPSIS
    final_df = df_clean[['Model', 'RAM', 'Storage', 'Berat', 'Harga']].dropna()
    # Remove rows with zero or negative values for meaningful comparison
    final_df = final_df[(final_df['RAM'] > 0) & (final_df['Storage'] >= 0) & (final_df['Berat'] > 0) & (final_df['Harga'] > 0)]
    return final_df

# Preprocess the data
with st.spinner("Memproses data..."):
    processed_data = preprocess_laptop_data(data)
    if processed_data is None:
        st.stop()
    st.success(f"✅ Data berhasil diproses! Tersedia {len(processed_data)} laptop alternatif.")

# ------------------------------------------------------------------------------
# Sidebar: Bobot dan orientasi kriteria
st.sidebar.subheader("🎚️ Bobot Kriteria")
# Bobot default sesuai makalah: RAM=0.4, Storage=0.2, Berat=0.15, Harga=0.25
w_ram = st.sidebar.slider("RAM (benefit)", 0.0, 1.0, 0.40, step=0.05)
w_storage = st.sidebar.slider("Storage (benefit)", 0.0, 1.0, 0.20, step=0.05)
w_berat = st.sidebar.slider("Berat (cost)", 0.0, 1.0, 0.15, step=0.05)
w_harga = st.sidebar.slider("Harga (cost)", 0.0, 1.0, 0.25, step=0.05)

# Normalisasi bobot
total_weight = w_ram + w_storage + w_berat + w_harga
if total_weight != 1.0:
    st.sidebar.warning(f"Total bobot = {total_weight:.2f}. Bobot akan dinormalisasi.")
    w_ram_norm = w_ram / total_weight
    w_storage_norm = w_storage / total_weight
    w_berat_norm = w_berat / total_weight
    w_harga_norm = w_harga / total_weight
else:
    w_ram_norm, w_storage_norm, w_berat_norm, w_harga_norm = w_ram, w_storage, w_berat, w_harga

weights = {
    'RAM': w_ram_norm,
    'Storage': w_storage_norm,
    'Berat': w_berat_norm,
    'Harga': w_harga_norm
}

st.sidebar.markdown("**Orientasi Kriteria (sesuai makalah):**")
st.sidebar.markdown("- RAM: Benefit (semakin besar semakin baik)")
st.sidebar.markdown("- Storage: Benefit")
st.sidebar.markdown("- Berat: Cost (semakin ringan semakin baik)")
st.sidebar.markdown("- Harga: Cost (semakin murah semakin baik)")

# ------------------------------------------------------------------------------
# Fungsi TOPSIS
def topsis(df, weights, benefit_criteria):
    """
    df: DataFrame dengan kolom: Model, RAM, Storage, Berat, Harga
    weights: dictionary bobot per kriteria
    benefit_criteria: list nama kriteria yang bersifat benefit
    """
    # Matriks keputusan (hanya nilai numerik)
    X = df[['RAM', 'Storage', 'Berat', 'Harga']].values.astype(float)
    m, n = X.shape
    
    # Langkah 1: Normalisasi Euclidean
    norm = np.sqrt((X**2).sum(axis=0))
    R = X / norm
    
    # Langkah 2: Normalisasi terbobot
    w = np.array([weights['RAM'], weights['Storage'], weights['Berat'], weights['Harga']])
    Y = R * w
    
    # Langkah 3: Solusi ideal positif dan negatif
    ideal_pos = []
    ideal_neg = []
    for j in range(n):
        if df.columns[j+1] in benefit_criteria:  # +1 karena kolom pertama 'Model'
            ideal_pos.append(np.max(Y[:, j]))
            ideal_neg.append(np.min(Y[:, j]))
        else:
            ideal_pos.append(np.min(Y[:, j]))
            ideal_neg.append(np.max(Y[:, j]))
    
    # Langkah 4: Jarak Euclidean
    D_pos = np.sqrt(((Y - ideal_pos)**2).sum(axis=1))
    D_neg = np.sqrt(((Y - ideal_neg)**2).sum(axis=1))
    
    # Langkah 5: Nilai preferensi
    V = D_neg / (D_pos + D_neg)
    
    # Tambahkan ke DataFrame
    result_df = df.copy()
    result_df['Skor_TOPSIS'] = V
    result_df['Rank'] = result_df['Skor_TOPSIS'].rank(ascending=False, method='min').astype(int)
    result_df = result_df.sort_values('Rank')
    return result_df

# ------------------------------------------------------------------------------
# Tombol hitung TOPSIS
if st.sidebar.button("🚀 Hitung & Tampilkan Ranking", use_container_width=True):
    benefit_cols = ['RAM', 'Storage']  # RAM dan Storage benefit
    with st.spinner("Menghitung TOPSIS ..."):
        ranking_df = topsis(processed_data, weights, benefit_cols)
    
    # Simpan ke session state agar bisa diakses di berbagai bagian
    st.session_state['ranking'] = ranking_df
    
    # Tampilkan hasil
    st.subheader("🏆 Hasil Perankingan Laptop (TOPSIS)")
    
    # Tampilkan top N slider
    top_n = st.slider("Tampilkan N alternatif teratas", min_value=5, max_value=min(50, len(ranking_df)), value=10, step=5)
    top_df = ranking_df.head(top_n)
    
    # Tabel rapi
    st.dataframe(
        top_df[['Rank', 'Model', 'RAM', 'Storage', 'Berat', 'Harga', 'Skor_TOPSIS']].style.format({
            'Skor_TOPSIS': '{:.4f}',
            'Harga': 'Rp {:,.0f}',
            'RAM': '{:.0f} GB',
            'Storage': '{:.0f} GB',
            'Berat': '{:.2f} kg'
        }),
        use_container_width=True
    )
    
    # Bar chart top 10
    st.subheader("📊 Visualisasi Skor Preferensi (Top 10)")
    bar_fig = px.bar(
        top_df.head(10), 
        x='Model', y='Skor_TOPSIS', 
        color='Skor_TOPSIS',
        text='Skor_TOPSIS',
        title="Skor TOPSIS per Laptop",
        labels={'Skor_TOPSIS': 'Nilai Preferensi', 'Model': 'Laptop'}
    )
    bar_fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    bar_fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(bar_fig, use_container_width=True)
    
    # Radar chart perbandingan top 5
    st.subheader("📡 Radar Chart Perbandingan Top 5 Laptop (Data Ternormalisasi)")
    # Ambil top 5
    top5 = ranking_df.head(5).copy()
    # Normalisasi min-max untuk setiap kriteria (agar skala 0-1, benefit = higher better, cost = lower better)
    # Lebih baik menggunakan nilai ternormalisasi asli sebelum pembobotan? Kita buat sendiri untuk radar.
    # Kita hitung nilai skala 0-1 (benefit: nilai asli/max, cost: min/nilai asli)
    X_radar = top5[['RAM', 'Storage', 'Berat', 'Harga']].copy()
    # Benefit
    X_radar['RAM'] = X_radar['RAM'] / X_radar['RAM'].max()
    X_radar['Storage'] = X_radar['Storage'] / X_radar['Storage'].max()
    # Cost (semakin kecil nilai asli, semakin baik -> skor = min/nilai)
    X_radar['Berat'] = X_radar['Berat'].min() / X_radar['Berat']
    X_radar['Harga'] = X_radar['Harga'].min() / X_radar['Harga']
    # Radar chart
    categories = ['RAM', 'Storage', 'Berat', 'Harga']
    fig_radar = go.Figure()
    for i, row in top5.iterrows():
        values = X_radar.loc[i, categories].tolist()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=row['Model'][:20]  # truncate name
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Perbandingan kriteria (skala 0-1, benefit & cost diolah)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Tampilkan alternatif terbaik
    best = ranking_df.iloc[0]
    st.success(f"✨ **Rekomendasi Terbaik:** {best['Model']} dengan skor preferensi {best['Skor_TOPSIS']:.4f}")
    
    # Penjelasan singkat
    with st.expander("ℹ️ Penjelasan Metode TOPSIS"):
        st.markdown("""
        **TOPSIS** (Technique for Order Preference by Similarity to Ideal Solution)  
        - Alternatif terbaik adalah yang memiliki jarak terpendek ke **Solusi Ideal Positif** dan jarak terjauh ke **Solusi Ideal Negatif**.  
        - Langkah-langkah:  
          1. Normalisasi matriks keputusan (Euclidean).  
          2. Pembobotan normalisasi.  
          3. Menentukan solusi ideal positif (A+) dan negatif (A-).  
          4. Menghitung jarak Euclidean ke A+ dan A-.  
          5. Menghitung nilai preferensi (V).  
          6. Perankingan berdasarkan V (semakin besar semakin baik).  
        - Kriteria yang digunakan: RAM (benefit), Storage (benefit), Berat (cost), Harga (cost).  
        - Bobot dapat disesuaikan sesuai preferensi pengguna di sidebar.
        """)
else:
    # Jika belum menghitung, tampilkan info
    st.info("👈 Atur bobot kriteria di sidebar, lalu tekan tombol 'Hitung & Tampilkan Ranking'.")
    st.markdown("""
    **Contoh penggunaan:**
    - Dataset berisi ribuan laptop dengan spesifikasi RAM, Storage, Berat, dan Harga.
    - Sistem akan meranking semua laptop menggunakan metode TOPSIS.
    - Anda dapat mengubah bobot kriteria sesuai kebutuhan (misalnya prioritas RAM lebih tinggi untuk keperluan data science, atau harga lebih murah untuk budget terbatas).
    """)
    if st.checkbox("Lihat pratinjau data (setelah preprocessing)"):
        st.dataframe(processed_data.head(20), use_container_width=True)

# ------------------------------------------------------------------------------
# Footer
st.markdown("---")
st.caption("Dibangun berdasarkan referensi makalah 'Analisis dan Implementasi Sistem Keputusan Cerdas untuk Pemilihan Laptop Menggunakan Metode TOPSIS' (2026).")